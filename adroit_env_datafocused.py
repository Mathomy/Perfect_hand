<<<<<<< HEAD
=======
import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco


class AdroitTrajEnv(gym.Env):
    """
    Adroit Hand environment for pinching task.
    Environment used to train pinch grasps with the index finger and thumb.
    Primarly designed to establish a 
    Key Differences from Reach:
    - Always starts from neutral position
    - Reward shaped for pinching motion (Trajectory more detailed for future analysis)
    - Better tracking of pinch quality (Different reward components)
    """


    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode=None, defaultsettings=True):
        super().__init__()
        self.render_mode = render_mode
        self.max_step=800
        self.current_steps = 0
        # Default position extracted from mujoco- used for resets when defaultsettings=True
        self.defaultpos = np.array([
            0.00084, 0.00012, 1.5e-06, 0.0085, 0.0084, 0.0083, 1.5e-06,
            0.0085, 0.0084, 0.0083, 1.5e-06, 0.0085, 0.0084, 0.0083, 0.0086,
            1.5e-06, 0.0085, 0.0084, 0.0083, -1.5e-05, 0.0086, 6.2e-06,
            0.00071, -0.0083, 0.1, -0.1, 0.023, 1, 3e-15, -7.6e-11, 2.2e-25
        ], dtype=np.float32)
        self.defaultsettings = defaultsettings
       

        # Load model
        model_path = os.path.join(os.path.dirname(__file__), "Adroit", "adroit_hand.xml")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Le fichier {model_path} est introuvable.")

        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        # Actuators for index and thumb
        self.index_actuators = [2, 3, 4, 5]
        self.thumb_actuators = [19,20, 21, 22, 23]
        self.used_actuators = self.index_actuators + self.thumb_actuators

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(len(self.used_actuators),),
            dtype=np.float32
        )

        # Observation space
        obs_dim = self.model.nq + 3
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

        # Body IDs for thumb and finger tips
        self.thumb_body_name = "thdistal"
        self.finger_body_name = "ffdistal"

        try:
            self.thumb_tip_id = mujoco.mj_name2id(
        self.model, mujoco.mjtObj.mjOBJ_SITE, "S_thtip"
    )
            self.index_tip_id = mujoco.mj_name2id(
        self.model, mujoco.mjtObj.mjOBJ_SITE, "S_fftip")
            self.thumb_body_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, self.thumb_body_name
            )
            self.finger_body_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, self.finger_body_name
            )
      
            
        except Exception as e:
            self.thumb_body_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, self.thumb_body_name
            )
            self.finger_body_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, self.finger_body_name
            )
            print("Problème avec les noms de bodies (thumb/finger). ")
            raise e


        # Target position above the palm
        self.target_pos = np.array([0.0, -0.10, 0.25], dtype=np.float32)

        self.np_random = np.random.RandomState()
        self.renderer = None
        if self.render_mode == "rgb_array":
            self.renderer = mujoco.Renderer(self.model)
        self.reward_params = {
        # coefficients principaux
        "pinch_coef": -20.0,            # multiplie dist_fingers
        "target_coef": -5.0,            # multiplie dist_to_target * pinch_quality
        "pinch_quality_scale": 10.0,    # used in exp(-scale * dist)

        # straightness (penalty weight) - positive numbers: penalty = - weight * (1 - ratio)
        "straight_weight_index": 3.0,
        "straight_weight_thumb": 7.0,   # less or more than index depending on desired importance

        # bonus thresholds (distance thresholds -> additive bonus)
        "bonus_thresh": [
            (0.040, 10.0),
            (0.025, 20.0),
            (0.018, 30.0)
        ],

        # extra bonuses when also close to target: (dist_thresh, target_thresh, bonus)
        "target_bonus": [
            (0.025, 0.23, 15.0),
            (0.018, 0.18, 30.0)
        ],

        # bonus for straight finger pinch (dist_thresh, straightness_ratio_thresh, bonus)
        "straightness_bonus": (0.018, 0.85, 10.0),
        "terminal_reward": 100.0
    }
        self.bonus_claimed = {
            "bonus_thresh": [False] * len(self.reward_params["bonus_thresh"]),
            "target_bonus": [False] * len(self.reward_params["target_bonus"]),
        }

    def _get_obs(self):
        """Get current observation."""
        qpos = self.data.qpos.ravel()
        return np.concatenate([qpos, self.target_pos])
    


    def _straightness_ratio(self, knuckle_body_name: str, middle_body_name: str, tip_pos: np.ndarray):
        """
    Compute straightness ratio for a finger described by knuckle -> middle -> tip.
    Returns ratio in [0,1] where 1.0 means perfectly straight.
    Safe: returns 1.0 on any failure.
        """
        try:
            kn_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, knuckle_body_name)
            mid_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, middle_body_name)

            kn_pos = self.data.xpos[kn_id].copy()
            mid_pos = self.data.xpos[mid_id].copy()

            seg1 = np.linalg.norm(mid_pos - kn_pos)
            seg2 = np.linalg.norm(tip_pos - mid_pos)
            total_seg = seg1 + seg2 + 1e-9

            direct = np.linalg.norm(tip_pos - kn_pos)

            ratio = float(direct / total_seg)
            # clamp
            ratio = max(0.0, min(1.0, ratio))
            return ratio
        except Exception:
            return 1.0


    def _compute_bonus(self, dist_fingers, dist_to_target, straight_idx, straight_th):
        """
        Aggregate bonuses in a modular way using reward_params.
        """
        cfg = self.reward_params
        bonus = 0.0

        # distance thresholds
        for i, (thresh, val) in enumerate(cfg["bonus_thresh"]):
            if dist_fingers < thresh and not self.bonus_claimed["bonus_thresh"][i]:
                bonus += val
                claimed =self.bonus_claimed["bonus_thresh"][i]
                self.bonus_claimed["bonus_thresh"][i] = True  # mark as claimed
                # print(f"bonus dist trhsolfd  : {bonus},{claimed} ")

        # extra target-based bonuses
        for i, (d_thresh, t_thresh, val) in enumerate(cfg["target_bonus"]):
            if dist_fingers < d_thresh and dist_to_target < t_thresh and not self.bonus_claimed["target_bonus"][i]:
                bonus += val
                claimed =self.bonus_claimed["target_bonus"][i]
                self.bonus_claimed["target_bonus"][i] = True  # mark as claimed
                # print(f"bonus target trhsolfd  : {val},{claimed} ")

            # straightness bonus (index and thumb blended: require index straightness by default)
            s_thresh, s_ratio_thresh, s_val = cfg["straightness_bonus"]
            # give straightness bonus only if fingers are close
            if dist_fingers < s_thresh and (straight_idx > s_ratio_thresh and straight_th > s_ratio_thresh):
                bonus += s_val
                # print(f"bonus straightness  : {val},")
        terminated = dist_fingers < 0.015 and dist_to_target <  0.146
        terminal_bonus = cfg.get("terminal_reward", 0.0) if terminated else 0.0
        bonus+=terminal_bonus
        # if bonus >0 :
        #     print(f"bonus : {bonus}")


        return bonus


    def _compute_reward(self):
        """
        Reward shaping for pinch with:
        - pinch distance term
        - target distance term (scaled by pinch_quality)
        - straightness penalties for index and thumb
        - structured bonuses
        """
        cfg = self.reward_params

        # --- tip positions (sites) ---
        try:
            thumb_tip_pos = self.data.site_xpos[self.thumb_tip_id].copy()
            index_tip_pos = self.data.site_xpos[self.index_tip_id].copy()
        except Exception:
            # safe fallback to body centers (shouldn't happen if sites exist)
            thumb_tip_pos = self.data.xpos[self.thumb_body_id].copy()
            index_tip_pos = self.data.xpos[self.finger_body_id].copy()

        # Distances
        dist_fingers = float(np.linalg.norm(thumb_tip_pos - index_tip_pos))
        mid_pos = 0.5 * (thumb_tip_pos + index_tip_pos)
        dist_to_target = float(np.linalg.norm(mid_pos - self.target_pos))

        # Pinch reward (distance-based)
        pinch_reward = cfg["pinch_coef"] * dist_fingers

        # Pinch quality scaling (sharp when very close)
        pinch_quality = float(np.exp(-cfg["pinch_quality_scale"] * dist_fingers))

        # Target reward (encourage moving mid-point toward target, but only effective when fingers are close)
        target_reward = cfg["target_coef"] * dist_to_target * pinch_quality

        # Straightness ratios for index and thumb
        # Index uses ffknuckle -> ffmiddle -> fftip
        straight_idx = self._straightness_ratio("ffknuckle", "ffmiddle", index_tip_pos)
        # Thumb: use thproximal -> thmiddle -> thdistal as segments (adjust names if you prefer other bodies)
        straight_th = self._straightness_ratio("thproximal", "thmiddle", thumb_tip_pos)

        # Straightness penalties (negative when fingers bent)
        straightness_penalty_idx = - cfg["straight_weight_index"] * (1.0 - straight_idx)
        straightness_penalty_th  = - cfg["straight_weight_thumb"] * (1.0 - straight_th)

        # Aggregate straightness reward
        straightness_reward = float(straightness_penalty_idx + straightness_penalty_th)

        # Bonuses
        bonus = float(self._compute_bonus(dist_fingers, dist_to_target, straight_idx, straight_th))

        # Total reward
        total_reward = float(
            pinch_reward
            + target_reward
            + straightness_reward
            + bonus
        )
        # print (pinch_reward,
        #      target_reward,
        #      straightness_reward,
        #      bonus)

        # For debugging / info you can store last measures as attributes or return them via info in step()
        # e.g. self.last_straight_idx = straight_idx

        return total_reward, dist_fingers, dist_to_target

    def reset(self, seed=None, options=None):
        """Reset environment - uses neutral position if defaultsettings=True."""
        super().reset(seed=seed)

        # Reset simulation
        mujoco.mj_resetData(self.model, self.data)

        if not hasattr(self, "defaultsettings") or not self.defaultsettings:
            # Petits bruits aléatoires autour de la position neutre
            self.data.qpos[:] = 0.05 * self.np_random.uniform(-1, 1, size=self.model.nq)
            self.data.qvel[:] = 0.01 * self.np_random.uniform(-1, 1, size=self.model.nv)
        else:
            # ALWAYS use neutral position for consistent training
            if len(self.defaultpos) != self.model.nq:
                raise ValueError(f"default_start_pos length {len(self.defaultpos)} != model.nq {self.model.nq}")
            else:
                self.data.qpos[:] = self.defaultpos
                self.data.qvel[:] = 0.0

        # Fixed target position
        self.target_pos = np.array([0.0, -0.10, 0.25], dtype=np.float32)
        self.current_steps = 0

        # Update physics
        mujoco.mj_forward(self.model, self.data)

        obs = self._get_obs()
        info = {}
        self.bonus_claimed = {
        "bonus_thresh": [False] * len(self.reward_params["bonus_thresh"]),
        "target_bonus": [False] * len(self.reward_params["target_bonus"]),
    }

        return obs, info

    def step(self, action):
        """Execute action and return transition."""
        action = np.clip(action, -1, 1)

        # Zero out all controls, then apply only to used actuators
        self.data.ctrl[:] = 0.0
        self.data.ctrl[self.used_actuators] = action

        # Simulate
        for _ in range(5):
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        #reward, dist_fingers, dist_to_target, straightness_ratio = self._compute_reward()
        reward, dist_fingers, dist_to_target= self._compute_reward()
        self.current_steps += 1
        # Success condition: excellent pinch
        terminated = dist_fingers < 0.015 and dist_to_target <0.146
        truncated = self.current_steps >= self.max_step

        info = {
            "dist_fingers": dist_fingers,
            "dist_to_target": dist_to_target,
            "reward": reward,
            "terminated":terminated,
            "truncated": truncated
        }
        # print(info)

        return obs, reward, terminated, truncated, info


    def render(self):
        """Render the environment."""
        if self.render_mode == "rgb_array" and self.renderer is not None:
            self.renderer.update_scene(self.data)
            img = self.renderer.render()
            return img

    def get_frame(self):
        """Get current frame as numpy array (H, W, 3)."""
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model)
        self.renderer.update_scene(self.data)
        return self.renderer.render()

    def close(self):
        """Nettoyer les ressources."""
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

    def utilis(self):
        """Debug: Print body names."""
        for i in range(self.model.nbody):
            print(i, self.model.body(i).name)

    def debug_actuators(self):
        """Debug: Print actuator names."""
        print("=== Liste des actuateurs ===")
        print("nu =", self.model.nu)
        for i in range(self.model.nu):
            print(i, self.model.actuator(i).name)
>>>>>>> origin/reward_with_tips

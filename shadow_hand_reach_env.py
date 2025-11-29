import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
import time


class AdroitHandReachEnv(gym.Env):
    """
    Environnement à partir de Adroit (à la place de Reach).

    Objectif :
    - Le pouce et un doigt sélectionné doivent approcher une cible située au-dessus de la paume.
    - Reward = - distance entre le point moyen (pouce + doigt) et la cible.
    """

    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.max_step=600
        self.current_steps = 0

        # Charger le modèle Mujoco de l'Adroit Hand
        model_path = os.path.join(os.path.dirname(__file__), "Adroit", "adroit_hand.xml")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Le fichier {model_path} est introuvable. ")
        

        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        # Contrôler tous les actuateurs disponibles
        self.index_actuators = [2, 3, 4, 5]    
        self.thumb_actuators = [19, 20, 21, 22, 23]  
        self.used_actuators = self.index_actuators + self.thumb_actuators

        self.action_space = spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(len(self.used_actuators),),
            dtype=np.float32
        )

        # Observation : qpos (positions des articulations) + target (3D)
        obs_dim = self.model.nq + 3
        self.observation_space = spaces.Box(low=-np.inf,high=np.inf,shape=(obs_dim,),dtype=np.float32)

        # On met des noms par défaut, à vérifier avec un script de listing
        self.thumb_body_name =  "thdistal"  # placeholder
        self.finger_body_name = "ffdistal"  # placeholder

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

        # Cible (au-dessus de la paume) - valeur approximative, à ajuster ensuite
        self.target_pos = np.array([0.0, -0.10, 0.25], dtype=np.float32)


        self.np_random = np.random.RandomState()
        self.renderer = None
        if self.render_mode == "rgb_array":
            self.renderer = mujoco.Renderer(self.model)
        

    # Helpers internes

    def _get_obs(self):
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
        for thresh, val in cfg["bonus_thresh"]:
            if dist_fingers < thresh:
                bonus += val

        # extra target-based bonuses
        for d_thresh, t_thresh, val in cfg["target_bonus"]:
            if dist_fingers < d_thresh and dist_to_target < t_thresh:
                bonus += val

        # straightness bonus (index and thumb blended: require index straightness by default)
        s_thresh, s_ratio_thresh, s_val = cfg["straightness_bonus"]
        # give straightness bonus only if fingers are close
        if dist_fingers < s_thresh and (straight_idx > s_ratio_thresh or straight_th > s_ratio_thresh):
            bonus += s_val

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

        # For debugging / info you can store last measures as attributes or return them via info in step()
        # e.g. self.last_straight_idx = straight_idx

        return total_reward, dist_fingers, dist_to_target


    # def _compute_reward(self): # Biggest changes between both environments are here
        
    #     #Reward function optimized for PINCHING motion.
        
    #     #Goals:
    #     #1. Bring thumb and index close together (pinch)
    #     #2. Keep fingers straight during pinching to have a better trajectory
    #     #3. Move the pinch point toward the target
    #     #4. Maintain proper finger alignment

    #     # Get finger tip positions
    #     thumb_tip_pos = self.data.site_xpos[self.thumb_tip_id].copy()
    #     index_tip_pos =self.data.site_xpos[self.index_tip_id].copy()

    #     # Distance between fingertips (pinch quality)
    #     dist_fingers = np.linalg.norm(thumb_tip_pos - index_tip_pos)

    #     # Midpoint between fingers
    #     mid_pos = 0.5 * (thumb_tip_pos + index_tip_pos)
        
    #     # Distance from midpoint to target
    #     dist_to_target = np.linalg.norm(mid_pos - self.target_pos)

    #     # Finger Straightness Penalty Calculation
    #     try:
    #         # Get body segment IDs of fingers
    #         ffknuckle_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "ffknuckle")
    #         ffmiddle_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "ffmiddle")
            
    #         # Get positions
    #         ff_knuckle = self.data.xpos[ffknuckle_id].copy()
    #         ff_middle = self.data.xpos[ffmiddle_id].copy()
    #         ff_tip = index_tip_pos
            
    #         # Measure straightness: compare actual distance to sum of segment lengths
    #         # Straighter finger = actual distance closer to sum of segments
    #         segment1 = np.linalg.norm(ff_middle - ff_knuckle)
    #         segment2 = np.linalg.norm(ff_tip - ff_middle)
    #         total_segments = segment1 + segment2
            
    #         direct_distance = np.linalg.norm(ff_tip - ff_knuckle)
            
    #         # Straightness ratio: 1.0 = perfectly straight, <1.0 = bent
    #         straightness_ratio = direct_distance / (total_segments + 1e-6)
            
    #         # Penalty for bent fingers
    #         straightness_penalty = -5.0 * (1.0 - straightness_ratio)
            
    #     except:
    #         straightness_penalty = 0.0
    #         straightness_ratio = 1.0

    #     # === REWARD COMPONENTS ===
        
    #     # Encourage pinching (fingers close together)
    #     pinch_reward = -20.0 * dist_fingers
        
    #     #Encourage straight fingers during pinch (to have a better trajectory)
    #     straightness_reward = straightness_penalty
        
       
    #     # Scale this by how close the fingers are
    #     pinch_quality = np.exp(-10 * dist_fingers)  # 1.0 when touching, ~0 when far
    #     target_reward = -5.0 * dist_to_target * pinch_quality
        
    #     # 4. Bonus rewards for achieving pinch
    #     bonus = 0.0
    #     if dist_fingers < 0.04:  # Starting to pinch
    #         bonus += 2.0
    #     if dist_fingers < 0.025:  # Good pinch
    #         bonus += 5.0
    #     if dist_fingers < 0.015:  # Excellent pinch
    #         bonus += 10.0
            
    #     # 5. Extra bonus if pinching AT the target location
    #     if dist_fingers < 0.025 and dist_to_target < 0.05:
    #         bonus += 15.0
    #     if dist_fingers < 0.015 and dist_to_target < 0.03:
    #         bonus += 25.0
            
    #     # 6. Additional bonus for straight finger pinch
    #     if dist_fingers < 0.025 and straightness_ratio > 0.85:
    #         bonus += 10.0

    #     total_reward = pinch_reward + straightness_reward + target_reward + bonus

    #     return total_reward, dist_fingers, dist_to_target
    # def _compute_reward(self):
    #     # Positions du pouce et de l'index dans le monde
    #     thumb_pos = self.data.site_xpos[self.thumb_tip_id].copy()
    #     index_pos = self.data.site_xpos[self.index_tip_id].copy()

    #     # Distance entre le pouce et l'index
    #     dist_fingers = np.linalg.norm(thumb_pos - index_pos)

    #     # Distance du milieu des deux doigts à la target
    #     mid_pos = 0.5 * (thumb_pos + index_pos)
    #     dist_to_target = np.linalg.norm(mid_pos - self.target_pos)

    #     # Reward :
    #     # - on veut rapprocher les doigts entre eux ET de la cible
    #     reward = 0.0
    #     reward += -10.0 * dist_fingers     # rapprocher pouce/index
    #     reward += -10.0 * dist_to_target   # rapprocher du point cible

    #     # Bonus si les doigts sont très proches
    #     if dist_fingers < 0.03:
    #         reward += 1.0
    #     if dist_fingers < 0.015:
    #         reward += 5.0

    #     return reward, dist_fingers, dist_to_target


    # API Gymnasium

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset Mujoco
        mujoco.mj_resetData(self.model, self.data)

        # Pose neutre
        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0

        # Cible FIXE au-dessus de la paume
        self.target_pos = np.array([0.0, -0.10, 0.25], dtype=np.float32)

    

        # Recalcul des positions
        mujoco.mj_forward(self.model, self.data)

        obs = self._get_obs()
        info = {}

        # # Lancer le viewer si besoin
        # if self.render_mode == "human" and self.viewer is None:
        #     self.viewer = mujoco.viewer.launch(self.model, self.data)
        self.current_steps = 0
        return obs, info

    
    # def step(self, action):
    #     # Clip des actions
    #     action = np.clip(action, self.action_space.low, self.action_space.high)

    #     # Mapping simple [-1,1] -> torque dans ctrl
    #     self.data.ctrl[:] = action

    #     # Avancer la simu
    #     n_substeps = 5
    #     for _ in range(n_substeps):
    #         mujoco.mj_step(self.model, self.data)

    #     obs = self._get_obs()
    #     reward, dist = self._compute_reward()

    #     # Terminaison si assez proche de la cible
    #     terminated = dist < 0.01
    #     truncated = False

    #     info = {"distance": dist}

    #     if self.render_mode == "human" and self.viewer is not None:
    #         self.viewer.sync()

    #     return obs, reward, terminated, truncated, info
    def step(self, action):

        action = np.clip(action, -1, 1)

        # Remettre à zéro toutes les autres articulations
        self.data.ctrl[:] = 0.0

        # Appliquer seulement les actions pour pouce + index
        self.data.ctrl[self.used_actuators] = action

        # Simulation
        for _ in range(5):
            mujoco.mj_step(self.model, self.data)


        obs = self._get_obs()
        # reward, dist = self._compute_reward()
        reward, dist_fingers, dist_to_target = self._compute_reward()

        #info = {"distance": dist, "reward": reward}
        info = {
        "dist_fingers": dist_fingers,
        "dist_to_target": dist_to_target,
        "reward": reward,
        }

        self.current_steps += 1

        terminated = dist_fingers < 0.015
        truncated = self.current_steps >= self.max_step
        print (terminated,truncated)
        # print("dist_fingers =", dist_fingers)

        return obs, reward, terminated, truncated, info
    # def step(self, action):
    #     action = np.clip(action, self.action_space.low, self.action_space.high)
    #     self.data.ctrl[:] = action

    #     n_substeps = 5
    #     for _ in range(n_substeps):
    #         mujoco.mj_step(self.model, self.data)

    #     obs = self._get_obs()
    #     reward, dist = self._compute_reward()
    #     terminated = dist < 0.01
    #     truncated = False
    #     info = {"distance": dist}

    #     # Synchroniser avec le temps réel
    #     if self.render_mode == "human" and self.viewer is not None:
    #         self.viewer.sync()
    #         time.sleep(1 / 60)  # 60 FPS

    #     return obs, reward, terminated, truncated, info
    

    def render(self):
        if self.render_mode == "rgb_array" and self.renderer is not None:
            self.renderer.update_scene(self.data)
            img = self.renderer.render()
            return img

    def close(self):
        # if self.viewer is not None:
        #     self.viewer.close()
        #     self.viewer = None
        pass
    def utilis(self) :
        for i in range(self.model.nbody):
            print(i, self.model.body(i).name)
    

    def debug_actuators(self): # Affiche la liste des actuateurs
        print("=== Liste des actuateurs ===")
        print("nu =", self.model.nu)
        for i in range(self.model.nu):
            print(i, self.model.actuator(i).name)

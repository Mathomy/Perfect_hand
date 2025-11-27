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

        # Default position extracted from mujoco- used for resets when defaultsettings=True
        self.defaultpos = np.array([
            0.00084, 0.00012, 1.5e-06, 0.0085, 0.0084, 0.0083, 1.5e-06,
            0.0085, 0.0084, 0.0083, 1.5e-06, 0.0085, 0.0084, 0.0083, 0.0086,
            1.5e-06, 0.0085, 0.0084, 0.0083, -1.5e-05, 0.0086, 6.2e-06,
            0.00071, -0.0083, 0.1, -0.1, 0.015, 1, 3e-15, -7.6e-11, 2.2e-25
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
        self.thumb_actuators = [19, 20, 21, 22, 23]
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
            self.thumb_body_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, self.thumb_body_name
            )
            self.finger_body_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, self.finger_body_name
            )
        except Exception as e:
            print("Problème avec les noms de bodies (thumb/finger).")
            raise e

        # Target position above the palm
        self.target_pos = np.array([0.0, -0.10, 0.25], dtype=np.float32)

        self.np_random = np.random.RandomState()
        self.renderer = None
        if self.render_mode == "rgb_array":
            self.renderer = mujoco.Renderer(self.model)

    def _get_obs(self):
        """Get current observation."""
        qpos = self.data.qpos.ravel()
        return np.concatenate([qpos, self.target_pos])
    

   
    def _compute_reward(self): # Biggest changes between both environments are here
        
        #Reward function optimized for PINCHING motion.
        
        #Goals:
        #1. Bring thumb and index close together (pinch)
        #2. Keep fingers straight during pinching to have a better trajectory
        #3. Move the pinch point toward the target
        #4. Maintain proper finger alignment

        # Get finger tip positions
        thumb_tip_pos = self.data.xpos[self.thumb_body_id].copy()
        index_tip_pos = self.data.xpos[self.finger_body_id].copy()

        # Distance between fingertips (pinch quality)
        dist_fingers = np.linalg.norm(thumb_tip_pos - index_tip_pos)

        # Midpoint between fingers
        mid_pos = 0.5 * (thumb_tip_pos + index_tip_pos)
        
        # Distance from midpoint to target
        dist_to_target = np.linalg.norm(mid_pos - self.target_pos)

        # Finger Straightness Penalty Calculation
        try:
            # Get body segment IDs of fingers
            ffknuckle_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "ffknuckle")
            ffmiddle_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "ffmiddle")
            
            # Get positions
            ff_knuckle = self.data.xpos[ffknuckle_id].copy()
            ff_middle = self.data.xpos[ffmiddle_id].copy()
            ff_tip = index_tip_pos
            
            # Measure straightness: compare actual distance to sum of segment lengths
            # Straighter finger = actual distance closer to sum of segments
            segment1 = np.linalg.norm(ff_middle - ff_knuckle)
            segment2 = np.linalg.norm(ff_tip - ff_middle)
            total_segments = segment1 + segment2
            
            direct_distance = np.linalg.norm(ff_tip - ff_knuckle)
            
            # Straightness ratio: 1.0 = perfectly straight, <1.0 = bent
            straightness_ratio = direct_distance / (total_segments + 1e-6)
            
            # Penalty for bent fingers
            straightness_penalty = -5.0 * (1.0 - straightness_ratio)
            
        except:
            straightness_penalty = 0.0
            straightness_ratio = 1.0

        # === REWARD COMPONENTS ===
        
        # Encourage pinching (fingers close together)
        pinch_reward = -20.0 * dist_fingers
        
        #Encourage straight fingers during pinch (to have a better trajectory)
        straightness_reward = straightness_penalty
        
       
        # Scale this by how close the fingers are
        pinch_quality = np.exp(-10 * dist_fingers)  # 1.0 when touching, ~0 when far
        target_reward = -5.0 * dist_to_target * pinch_quality
        
        # 4. Bonus rewards for achieving pinch
        bonus = 0.0
        if dist_fingers < 0.04:  # Starting to pinch
            bonus += 2.0
        if dist_fingers < 0.025:  # Good pinch
            bonus += 5.0
        if dist_fingers < 0.015:  # Excellent pinch
            bonus += 10.0
            
        # 5. Extra bonus if pinching AT the target location
        if dist_fingers < 0.025 and dist_to_target < 0.05:
            bonus += 15.0
        if dist_fingers < 0.015 and dist_to_target < 0.03:
            bonus += 25.0
            
        # 6. Additional bonus for straight finger pinch
        if dist_fingers < 0.025 and straightness_ratio > 0.85:
            bonus += 10.0

        total_reward = pinch_reward + straightness_reward + target_reward + bonus

        return total_reward, dist_fingers, dist_to_target
    """ 
    def _compute_reward(self):
        # Positions du pouce et de l'index dans le monde
        thumb_pos = self.data.xpos[self.thumb_body_id].copy()
        index_pos = self.data.xpos[self.finger_body_id].copy()

        # Distance entre le pouce et l'index
        dist_fingers = np.linalg.norm(thumb_pos - index_pos)

        # Distance du milieu des deux doigts à la target
        mid_pos = 0.5 * (thumb_pos + index_pos)
        dist_to_target = np.linalg.norm(mid_pos - self.target_pos)

        # Reward :
        # - on veut rapprocher les doigts entre eux ET de la cible
        reward = 0.0
        reward += -10.0 * dist_fingers     # rapprocher pouce/index
        reward += -10.0 * dist_to_target   # rapprocher du point cible

        # Bonus si les doigts sont très proches
        if dist_fingers < 0.03:
            reward += 1.0
        if dist_fingers < 0.015:
            reward += 5.0

        return reward, dist_fingers, dist_to_target
     """
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

        # Update physics
        mujoco.mj_forward(self.model, self.data)

        obs = self._get_obs()
        info = {}

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
        # Success condition: excellent pinch
        terminated = dist_fingers < 0.015
        truncated = False

        info = {
            "dist_fingers": dist_fingers,
            "dist_to_target": dist_to_target,
            "reward": reward,
        }

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
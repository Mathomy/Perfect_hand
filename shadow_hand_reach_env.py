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
    FINGER_CONFIG = {
        "thumb":  {"actuators": [19, 20, 21, 22, 23], "body": "thdistal"},
        "index":  {"actuators": [2, 3, 4, 5],     "body": "ffdistal"},
        "middle": {"actuators": [6, 7, 8, 9],     "body": "mf2"},
        "ring":   {"actuators": [10, 11, 12, 13], "body": "rf2"},
        "pinky":  {"actuators": [14, 15, 16, 17], "body": "lf2"},
    }

    def __init__(self, render_mode=None, fingers=None, target_fingers=("thumb", "index")):
        super().__init__()

        self.render_mode = render_mode

        # --- doigts actifs ---
        if fingers is None:
            fingers = ["thumb", "index"]
        self.fingers = fingers
        self.target_fingers = target_fingers

        # --- actuators utilisés ---
        self.used_actuators = []
        for f in self.fingers:
            self.used_actuators += self.FINGER_CONFIG[f]["actuators"]
        self.used_actuators = sorted(self.used_actuators)

        # --- action space ---
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(len(self.used_actuators),),
            dtype=np.float32
        )

        # --- chargement modèle MuJoCo ---
        model_path = os.path.join(os.path.dirname(__file__), "Adroit", "adroit_hand.xml")
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        # --- body IDs ---
        self.body_ids = {}
        for f in self.fingers:
            body_name = self.FINGER_CONFIG[f]["body"]
            self.body_ids[f] = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)

        # --- target ---
        self.target_pos = np.array([0.0, -0.10, 0.25], dtype=np.float32)

    # Helpers internes

    def _get_obs(self):
        qpos = self.data.qpos.ravel()
        return np.concatenate([qpos, self.target_pos])


    def _compute_reward(self):
        f1, f2 = self.target_fingers

        pos1 = self.data.xpos[self.body_ids[f1]]
        pos2 = self.data.xpos[self.body_ids[f2]]

        dist_fingers = np.linalg.norm(pos1 - pos2)

        # Milieu
        mid = 0.5 * (pos1 + pos2)
        dist_target = np.linalg.norm(mid - self.target_pos)

        # Reward générique
        reward = -10 * dist_fingers - 10 * dist_target

        # Bonus
        if dist_fingers < 0.03:
            reward += 1.0
        if dist_fingers < 0.015:
            reward += 5.0

        return reward, dist_fingers, dist_target

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
        return obs, info
    

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
        terminated = dist_fingers < 0.015
        truncated = False

        #info = {"distance": dist, "reward": reward}
        info = {
        "dist_fingers": dist_fingers,
        "dist_to_target": dist_to_target,
        "reward": reward,
        }

        return obs, reward, terminated, truncated, info
    

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

import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco.viewer
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

        # Charger le modèle Mujoco de l'Adroit Hand
        model_path = os.path.join(os.path.dirname(__file__), "Adroit", "adroit_hand.xml")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Le fichier {model_path} est introuvable. "
            )
        

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
        self.thumb_body_name = "fftip"   # placeholder
        self.finger_body_name = "thtip"  # placeholder

        try:
            self.thumb_body_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, self.thumb_body_name
            )
            self.finger_body_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, self.finger_body_name
            )
        except Exception as e:
            print("Problème avec les noms de bodies (thumb/finger). ")
            raise e

        # Cible (au-dessus de la paume) - valeur approximative, à ajuster ensuite
        self.target_pos = np.zeros(3, dtype=np.float32)


        self.viewer = None

    # Helpers internes

    def _get_obs(self):
        qpos = self.data.qpos.ravel()
        return np.concatenate([qpos, self.target_pos])

    def _compute_reward(self):
        thumb_pos = self.data.xpos[self.thumb_body_id].copy()
        index_pos = self.data.xpos[self.finger_body_id].copy()

        dist = np.linalg.norm(thumb_pos - index_pos)

        reward = -dist
        reward += 1.0 / (dist + 0.01)  # bonus dense
        if dist < 0.015:
            reward += 5.0  # succès

        # Optionnel : target dynamique
        self.target_pos = 0.5 * (thumb_pos + index_pos) + np.array([0,0,0.01])

        return reward, dist

    # API Gymnasium

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset Mujoco
        mujoco.mj_resetData(self.model, self.data)

        # Pose neutre
        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0

        # Nouvelle cible
        self._update_target()
    

        # Recalcul des positions
        mujoco.mj_forward(self.model, self.data)

        obs = self._get_obs()
        info = {}

        # Lancer le viewer si besoin
        if self.render_mode == "human" and self.viewer is None:
            self.viewer = mujoco.viewer.launch(self.model, self.data)
        return obs, info
    def _update_target(self):
    # Assure que np_random est défini
        if not hasattr(self, "np_random"):
            self.np_random = np.random.RandomState()

        self.target_pos = np.array([
            0.05 * self.np_random.uniform(-1, 1),
            -0.10 + 0.05 * self.np_random.uniform(-1, 1),
            0.25 + 0.05 * self.np_random.uniform(-1, 1),
        ], dtype=np.float32)
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
        self.data.ctrl[:] = 0

        # Appliquer seulement les actions pour pouce + index
        self.data.ctrl[self.used_actuators] = action

        # Simulation
        for _ in range(5):
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        reward, dist = self._compute_reward()
        terminated = dist < 0.015
        truncated = False

        return obs, reward, terminated, truncated, {"distance": dist}
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
        pass  # déjà géré par le viewer

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
    def utilis(self) :
        for i in range(self.model.nbody):
            print(i, self.model.body(i).name)
    

    def debug_actuators(self): # Affiche la liste des actuateurs
        print("=== Liste des actuateurs ===")
        print("nu =", self.model.nu)
        for i in range(self.model.nu):
            print(i, self.model.actuator(i).name)

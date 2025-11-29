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


    def _compute_reward(self):
        # Positions du pouce et de l'index dans le monde
        thumb_pos = self.data.xpos[self.thumb_body_id].copy()
        index_pos = self.data.xpos[self.finger_body_id].copy()

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

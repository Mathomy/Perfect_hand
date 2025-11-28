import time
import numpy as np
import mujoco

from stable_baselines3 import SAC
from stable_baselines3.common.buffers import ReplayBuffer

from shadow_hand_reach_env import AdroitHandReachEnv


class Explore:
    """
    Classe qui implémente Algorithm 1 (EXPLORE) de l'article :

    1. Initialise Q, pi (via SAC) et le replay buffer B.
    2. Pour chaque iteration:
       - pour chaque timestep:
           * collecter s_t, a_t, s_{t+1}
           * calculer r_int(s_t) = log( || s_t - s_t^{(k-NN)} || )
           * stocker (s_t, a_t, s_{t+1}, r_int) dans B
       - pour chaque gradient step:
           * prendre un minibatch dans B
           * mettre à jour SAC (critic et policy)

    Ici, on utilise stable-baselines3.SAC pour gérer les updates.
    Nous, on s'occupe de fournir le reward intrinsèque à chaque step.
    """

    def __init__(
        self,
        total_steps=1000,
        k=5,
        state_buffer_max_size=5000,
        model_path="explore.zip",
    ):
        # Hyperparamètres
        self.total_steps = total_steps
        self.k = k
        self.max_state_buffer = state_buffer_max_size
        self.model_path = model_path

        # Environnement
        self.env = AdroitHandReachEnv(render_mode=None)
        obs, info = self.env.reset()
        self.obs = obs

        # Dimensions
        self.obs_dim = self.env.observation_space.shape[0]
        self.act_dim = self.env.action_space.shape[0]

        # Buffer d'états pour l'estimateur k-NN de l'entropie
        self.state_buffer = []
        self.eps = 1e-6

        # Replay buffer au sens RL (transitions)
        self.model = SAC(
            policy="MlpPolicy",
            env=self.env,
            verbose=1,
            tensorboard_log="./sac_explore_tb/",
        )

    
    def add_state_to_buffer(self, s: np.ndarray):
        s = np.asarray(s, dtype=np.float32)
        self.state_buffer.append(s)
        if len(self.state_buffer) > self.max_state_buffer:
            self.state_buffer.pop(0)

    def knn_distance(self, s: np.ndarray, k: int = None) -> float:
        if k is None:
            k = self.k
        if len(self.state_buffer) < k:
            return 0.0

        s = np.asarray(s, dtype=np.float32)
        arr = np.stack(self.state_buffer, axis=0)
        diff = arr - s[None, :]
        dists = np.linalg.norm(diff, axis=1)
        dists_sorted = np.sort(dists)
        return float(dists_sorted[k - 1])

    def intrinsic_reward(self, s: np.ndarray) -> float:
        d_k = self.knn_distance(s, self.k)
        return float(np.log(d_k + self.eps))
    
    def run_exploration(self):
        """
        Boucle principale d'exploration (corrigée):
        - n'ajoute pas d'énormes r_int négatifs quand pas assez d'échantillons
        - appelle self.model.train(batch_size=..., gradient_steps=...)
        """
        # Initialiser le buffer d'états avec le premier état
        s0 = self.env.get_state_from_obs(self.obs)
        self.add_state_to_buffer(s0)

        rb = self.model.replay_buffer

        # Optionnel : préremplir le buffer d'états avec quelques actions aléatoires
        # pour éviter d'avoir d_k == 0 au début. (facultatif mais conseillé)
        prefill_steps = 200
        for _ in range(prefill_steps):
            a_rand = self.env.action_space.sample()
            obs_next, _, terminated, truncated, _ = self.env.step(a_rand)
            s_next = self.env.get_state_from_obs(obs_next)
            self.add_state_to_buffer(s_next)
            if terminated or truncated:
                obs_reset, _ = self.env.reset()

        # Boucle principale
        for step in range(self.total_steps):
            # 1) choisir action via la policy actuelle
            action, _ = self.model.predict(self.obs, deterministic=False)

            # 2) step env
            obs_next, r_ext, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            # 3) états s_t et s_next
            s_t = self.env.get_state_from_obs(self.obs)
            s_next = self.env.get_state_from_obs(obs_next)

            # 4) calcul du r_int, mais gérer le cas d_k == 0
            d_k = self.knn_distance(s_t)
            if d_k <= 0:
                # Pas assez de voisins / trop proche => reward intrinsèque neutre
                r_int = 0.0
            else:
                r_int = float(np.log(d_k + self.eps))

            # 5) mise à jour du buffer d'états
            self.add_state_to_buffer(s_t)

            # 6) stocker transition dans le replay buffer de SB3
            # infos doit être une liste contenant un dict (même si un seul env)
            infos = [info if isinstance(info, dict) else {}]
            rb.add(
                obs=self.obs,
                next_obs=obs_next,
                action=action,
                reward=r_int,
                done=done,
                infos=infos,
            )

            # 7) mettre à jour l'observation courante
            self.obs = obs_next

            # 8) reset si épisode terminé
            if done:
                obs_reset, info_reset = self.env.reset()
                self.obs = obs_reset
                # ré-initialiser le buffer d'états si tu le souhaites :
                self.add_state_to_buffer(self.env.get_state_from_obs(self.obs))

            # 9) faire des gradient steps (ici 1 step par interaction; tu peux augmenter)
            if rb.size() > self.model.batch_size:
                # gradient_steps = 1 (faire 1 step de mise à jour)
                self.model.train(batch_size=self.model.batch_size, gradient_steps=1)

            # 10) debug print
            if step % 1000 == 0:
                print(f"[step {step}] r_int = {r_int:.4f}, d_k = {d_k:.6f}, rb.size = {rb.size()}")


    # # Exploration
    # def run_exploration(self):
    #     """
    #     Boucle principale d'exploration :
    #     - collecte de transitions,
    #     - calcul r_int,
    #     - apprentissage SAC.
    #     """

    #     # Initialiser le buffer d'états avec le premier état
    #     s0 = self.env.get_state_from_obs(self.obs)
    #     self.add_state_to_buffer(s0)


    #     # Pour utiliser le replay buffer interne de SB3 :
    #     rb = self.model.replay_buffer
    #     rollout_buffer_size = self.total_steps

    #     # Pour chaque timestep
    #     for step in range(self.total_steps):
    #         #choisir action a_t depuis la policy actuelle
    #         action, _ = self.model.predict(self.obs, deterministic=False)

    #         # step env pour obtenir obs_next, r_ext, etc.
    #         obs_next, r_ext, terminated, truncated, info = self.env.step(action)
    #         done = terminated or truncated

    #         # calculer état s_t et s_{t+1}
    #         s_t = self.env.get_state_from_obs(self.obs)
    #         s_next = self.env.get_state_from_obs(obs_next)

    #         # calculer reward intrinsèque r_int(s_t)
    #         r_int = self.intrinsic_reward(s_t)

    #         # ajouter s_t dans le buffer d'états (APRES calcul)
    #         self.add_state_to_buffer(s_t)

    #         # stocker transition dans le replay buffer RL avec r_int
    #         if not isinstance(info, dict):
    #             info = {}
    #         rb.add(
    #             obs=self.obs,
    #             next_obs=obs_next,
    #             action=action,
    #             reward=r_int,
    #             done=done,
    #             infos=[info],
    #         )

    #         self.obs = obs_next

    #         # si épisode terminé, reset env
    #         if done:
    #             obs_reset, info_reset = self.env.reset()
    #             self.obs = obs_reset
    #             s0 = self.env.get_state_from_obs(self.obs)
    #             self.add_state_to_buffer(s0)

    #         # On attend d'avoir un minimum de transitions dans le replay buffer
    #         if rb.size() > self.model.batch_size:
    #             self.model.train(batch_size=self.model.batch_size)

    #         # Affichage debug
    #         if step % 1000 == 0:
    #             print(f"[step {step}] r_int = {r_int:.4f}, d_k = {self.knn_distance(s_t):.4f}")

    #     # Fin entraînement
    #     self.model.save(self.model_path)
    #     print(f"Pré-entraînement d'exploration terminé. Modèle sauvegardé : {self.model_path}")

    # Visualisation avec viewer Mujoco
    def visualize(self, n_steps=2000):
        """
        Exécute la policy apprise dans l'env avec un viewer Mujoco.
        """
        # Recharger proprement le modèle avec l'env
        self.model = SAC.load(self.model_path, env=self.env)

        obs, info = self.env.reset()

        with mujoco.viewer.launch_passive(self.env.model, self.env.data) as viewer:
            try:
                for t in range(n_steps):
                    action, _ = self.model.predict(obs, deterministic=False)
                    obs, reward, terminated, truncated, info = self.env.step(action)

                    viewer.sync()
                    time.sleep(1 / 60.0)

                    if terminated or truncated:
                        obs, info = self.env.reset()

            except KeyboardInterrupt:
                print("Visualisation interrompue (Ctrl+C).")

        self.env.close()

if __name__ == "__main__":
    explorer = Explore(
        total_steps=1000,
        k=5,
        state_buffer_max_size=5000,
        model_path="explore.zip",
    )

    explorer.run_exploration()
    explorer.visualize(n_steps=1000)

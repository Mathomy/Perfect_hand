
import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
import math
import mujoco
import mujoco.viewer
import time

# Importer l'environnement de départ
from shadow_hand_reach_env import AdroitHandReachEnv


class Std:
    def __init__(self, eps=1e-8):
        self.mean = 0.0
        self.var = 1.0
        self.count = 0
        self.eps = eps

    def update(self, x):
        x = float(x)
        self.count += 1
        if self.count == 1:
            self.mean = x
            self.var = 0.0
        else:
            old_mean = self.mean
            self.mean = old_mean + (x - old_mean) / self.count
            self.var = ((self.count - 2) * self.var + (x - old_mean) * (x - self.mean)) / (self.count - 1)
        return self.std()

    def std(self):
        return math.sqrt(self.var + self.eps)


class IntrinsicRewardWrapper(gym.Wrapper):
    def __init__(self, env, state_key='qpos', k=5, buffer_size=5000, normalize=True, eps=1e-6):
        """
        env: l'environnement original (AdroitHandReachEnv)
        state_key: 'qpos' to use self.data.qpos, or 'obs' to use full observation
        k: k-th nearest neighbor to use (k-NN)
        buffer_size: maximum number of states stored (FIFO)
        normalize: whether to normalize r_int by running std
        """
        super().__init__(env)
        self.k = max(1, k)
        self.buffer_size = buffer_size
        self.state_key = state_key
        self.eps = eps
        self.normalize = normalize

        # store states as numpy array list
        self.state_buffer = []  # list of 1D arrays
        self.running_std = Std()

        # Keep original action/obs/spaces
        self.action_space = env.action_space
        self.observation_space = env.observation_space

    def _get_state_vector(self, obs):
        """
        Choose what vector to treat as 'state' for k-NN:
         - if state_key == 'qpos' -> use env.data.qpos.ravel()
         - if state_key == 'obs' -> use the observation vector returned by env
        """
        if self.state_key == 'qpos':
            # Use the underlying MuJoCo qpos vector; env.data.qpos should exist
            s = self.env.data.qpos.ravel().copy()
        else:
            # obs provided by env.reset()/env.step
            s = np.array(obs).ravel().copy()
        return s

    def _compute_kth_dist(self, s):
        """
        Compute the k-th nearest neighbor distance between s and all states in buffer.
        If buffer smaller than k: return min distance.
        """
        if len(self.state_buffer) == 0:
            return None
        # stack to array (N, D)
        arr = np.vstack(self.state_buffer)  # might be heavy; buffer_size modest
        # Compute squared distances
        # Efficient enough for moderate buffer_size (<= 5000)
        dists = np.linalg.norm(arr - s[None, :], axis=1)
        # sort
        if len(dists) <= self.k:
            kth = np.min(dists)
        else:
            kth = np.partition(dists, self.k)[self.k]  # approximate k-th
        return float(kth)

    def _add_state_to_buffer(self, s):
        if len(self.state_buffer) >= self.buffer_size:
            # FIFO: pop first
            self.state_buffer.pop(0)
        self.state_buffer.append(s.copy())

    def _intrinsic_reward(self, s):
        """
        r_int = log( dist_k + eps ). Normalize by running std if desired.
        """
        kth = self._compute_kth_dist(s)
        if kth is None:
            # first samples get zero intrinsic reward but still added to buffer
            r = 0.0
        else:
            r = math.log(kth + self.eps)
        # update running std
        if self.normalize:
            self.running_std.update(r)
            std = self.running_std.std()
            if std > 1e-8:
                r_norm = r / std
            else:
                r_norm = r
        else:
            r_norm = r

        return float(r_norm)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        s = self._get_state_vector(obs)
        self._add_state_to_buffer(s)
        return obs, info

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        s = self._get_state_vector(obs)
        r_int = self._intrinsic_reward(s)
        # store state after computing intrinsic reward
        self._add_state_to_buffer(s)

        # return obs and intrinsic reward (SAC will use this)
        return obs, r_int, terminated, truncated, info

def train_exploration_sac(total_timesteps=100000, k=5, buffer_size=2000, seed=0):
    """
    Entrainer SAC sur la reward intrinsèque (exploration).
    - wrapped_env = IntrinsicRewardWrapper(AdroitHandReachEnv(), ...)
    - SAC learns to maximize intrinsic reward -> explore
    """
    env_fn = lambda: IntrinsicRewardWrapper(AdroitHandReachEnv(render_mode=None),
                                            state_key='qpos', k=k, buffer_size=buffer_size)
    venv = DummyVecEnv([env_fn])

    # instantiate SAC
    model = SAC("MlpPolicy", venv, verbose=1, seed=seed,
                buffer_size=100000, learning_starts=1000, batch_size=256)

    # Train
    model.learn(total_timesteps=total_timesteps)

    # Save model
    model.save("sac_explore_adroit")

    # Close
    venv.close()
    print("Exploration pre-training terminé. Modèle sauvegardé : sac_explore_adroit.zip")
    return model


def visualize_policy(model_path="sac_explore_adroit.zip", n_steps=1000):
    env = AdroitHandReachEnv(render_mode=None)
    model = SAC.load(model_path, env=env)

    
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        obs, info = env.reset()
        try:
            for t in range(n_steps):
                action, _ = model.predict(obs, deterministic=False)
                obs, reward, terminated, truncated, info = env.step(action)
                time.sleep(0.05)
                # reward here is extrinsic (env default) — if you want to see intrinsic, wrap as above
                viewer.sync()
        except KeyboardInterrupt:
            pass
    env.close()


if __name__ == "__main__":
    # Exemple : entraînement court pour debug
    model = train_exploration_sac(total_timesteps=5000, k=5, buffer_size=20, seed=1)
    # Visualiser quelques seconds
    visualize_policy(model_path="sac_explore_adroit.zip", n_steps=1000)

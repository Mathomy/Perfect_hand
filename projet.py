from shadow_hand_reach_env import AdroitHandReachEnv
from stable_baselines3 import PPO
from adroit_env_datafocused import AdroitTrajEnv
import time
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

# importe ton envs et ton callback
from shadow_hand_reach_env import AdroitHandReachEnv
from videocallback import LoggingVideoCallback  # ou le nom où tu as défini le callback

def train_ppo(total_timesteps=200_000,
              save_dir="logs",
              tensorboard_log="./ppo_shadowhand/",
              video_freq=20,
              frame_skip=2,
              device="cpu"):
    os.makedirs(save_dir, exist_ok=True)

    # --- ENV avec Monitor pour logging des rewards ---
    env_train = DummyVecEnv([lambda: Monitor(AdroitTrajEnv(render_mode="rgb_array", defaultsettings=True))])

    # --- CALLBACK (vidéo + entropies) ---
    callback = LoggingVideoCallback(
        env=env_train,
        save_dir=save_dir,
        video_freq=video_freq,
        save_success_videos=True,
        frame_skip=frame_skip,
        verbose=1
    )

    # --- MODEL ---
    model = PPO(
        policy="MlpPolicy",
        env=env_train,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
        tensorboard_log=tensorboard_log,
        device=device
    )

    # --- TRAIN ---
    model.learn(total_timesteps=total_timesteps, callback=callback)

    # --- SAVE MODEL & METRICS ---
    model_path = os.path.join(save_dir, "ppo_shadowhand_model_new_reward")
    model.save(model_path)
    print(f"✔ Model saved to: {model_path}")

    metrics_file = os.path.join(save_dir, "metrics_dataset.pkl")
    with open(metrics_file, "wb") as f:
        pickle.dump(callback.dataset, f)
    print(f"✔ Metrics saved to: {metrics_file}")

    # --- Compute random-policy entropy a---
    action_dim = env_train.action_space.shape[0]
    random_entropy = action_dim * np.log(2.0)  # uniform [-1,1] per action dim
    print(f"Random (uniform) policy entropy (nats) = {random_entropy:.4f} (action_dim={action_dim})")

    # --- Prepare plotting data ---
    episode_returns = callback.dataset.get("episode_returns", [])
    episode_lengths = callback.dataset.get("episode_lengths", [])
    per_episode_entropies = callback.dataset.get("entropies", [])
    mean_entropy_per_episode = [np.mean(ep) if len(ep) > 0 else np.nan for ep in per_episode_entropies]

    # --- PLOT RESULTS ---
    plt.figure(figsize=(14, 10))

    # 1) Episode rewards
    plt.subplot(3, 1, 1)
    if episode_returns:
        plt.plot(episode_returns, label="Episode return")
        plt.xlabel("Episode")
        plt.ylabel("Return")
        plt.title("Episode returns")
        plt.grid(True)
    else:
        plt.text(0.2, 0.5, "No episode returns logged", fontsize=12)

    # 2) Episode lengths
    plt.subplot(3, 1, 2)
    if episode_lengths:
        plt.plot(episode_lengths, label="Episode length")
        plt.xlabel("Episode")
        plt.ylabel("Length")
        plt.title("Episode lengths")
        plt.grid(True)
    else:
        plt.text(0.2, 0.5, "No episode lengths logged", fontsize=12)

    # 3) Policy entropy
    plt.subplot(3, 1, 3)
    if mean_entropy_per_episode:
        plt.plot(mean_entropy_per_episode, label="PPO mean entropy / episode")
    plt.axhline(y=random_entropy, color="k", linestyle="--", linewidth=2, label=f"random uniform entropy = {random_entropy:.3f} nats")
    plt.xlabel("Episode")
    plt.ylabel("Entropy (nats)")
    plt.title("Policy entropy per episode (PPO) vs Random uniform")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    env_train.close()

    return model, callback.dataset, random_entropy


# def train_ppo():
#     # --- Entraînement SANS rendu ---
#     env_train = AdroitHandReachEnv(render_mode=None)

#     model = PPO(
#         policy="MlpPolicy",
#         env=env_train,
#         verbose=1,
#         tensorboard_log="./ppo_shadowhand/",
#         device="cpu"
#     )

#     model.learn(total_timesteps=200_000)

#     model.save("ppo_shadowhand_new_rewardtips")
#     env_train.close()
#     print("✔ Entraînement terminé et modèle sauvegardé.")


# def evaluate_model():
#     # --- Évaluation AVEC rendu ---
#     env_eval = AdroitHandReachEnv(render_mode="human")
#     env_eval.utilis()
#     env_eval.debug_actuators()
#     model = PPO.load("ppo_shadowhand.zip", env=env_eval)

#     obs, info = env_eval.reset()
#     try : 
#         for _ in range(2000):
#             action, _ = model.predict(obs, deterministic=True)
#             print("action avant:", action)
#             obs, reward, terminated, truncated, info = env_eval.step(action)
#             print("action après", env_eval.data.qpos)

#             if terminated or truncated:
#                 obs, info = env_eval.reset()
#     finally :
#         env_eval.close()
import mujoco
import numpy as np
from mujoco import viewer

def visualize_trained_model():
    env_train = AdroitTrajEnv(render_mode="rgb_array", defaultsettings=True)
    # Charge le modèle entraîné
    model = PPO.load("C:/Users/tlamy/Sorbonne/Social robotic/Perfect_hand/logs/ppo_shadowhand_model_new_reward.zip", env=env_train)

    obs, info = env_train.reset()

    # Ouvre le viewer passif (géré par nous)
    with mujoco.viewer.launch_passive(env_train.model, env_train.data) as v:
        try:
            while True:  # boucle infinie jusqu'à Ctrl+C
                # Action du modèle
                time.sleep(0.3)
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env_train.step(action)

                # Debug simple si tu veux voir ce qui se passe
                # print("dist:", info.get("distance"), "reward:", reward)

                # Afficher la frame
                v.sync()
                print(terminated,truncated)
                # Si épisode fini → on reset mais on ne ferme PAS la fenêtre
                if terminated or truncated:
                    obs, info = env_train.reset()

        except KeyboardInterrupt:
            print("Visualisation interrompue par l'utilisateur (Ctrl+C).")

    env.close()

# def visualize_trained_model():
#     """Affiche la main qui bouge avec le modèle entraîné, via mujoco.viewer."""
#     env = AdroitHandReachEnv(render_mode=None)
#     model = PPO.load("ppo_shadowhand.zip", env=env)

#     obs, info = env.reset()

#     # # Viewer passif : on contrôle la simu nous-mêmes
#     # with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
#     #     for _ in range(1000):
#     #         action, _ = model.predict(obs, deterministic=True)
#     #         obs, reward, terminated, truncated, info = env.step(action)
#     #         print("dist:", info["distance"], "reward:", reward)
#     #         # Affiche un peu ce qui se passe dans le terminal si tu veux
#     #         # print("action:", action, "distance:", info["distance"])

#     #         viewer.sync()

#     #         if terminated or truncated:
#     #             obs, info = env.reset()
#     with viewer.launch_passive(env.model, env.data) as v:
#         for _ in range(1000):
#             action, _ = model.predict(obs, deterministic=True)
#             obs, reward, terminated, truncated, info = env.step(action)

#             print(
#                 "dist_fingers:", info["dist_fingers"],
#                 "dist_to_target:", info["dist_to_target"],
#                 "reward:", reward
#             )

#             v.sync()

#             if terminated or truncated:
#                 obs, info = env.reset()


#     env.close()

import pickle
import matplotlib.pyplot as plt
import numpy as np

def plot_metrics(metrics_file):
    # --- Charger le dataset ---
    with open(metrics_file, "rb") as f:
        dataset = pickle.load(f)

    # --- Extraire les métriques ---
    episode_returns = dataset.get("episode_returns", [])
    episode_lengths = dataset.get("episode_lengths", [])

    # Rewards cumulés par épisode (recalculé à partir des rewards et terminals si dispo)
    if all(key in dataset for key in ["rewards", "terminals"]):
        rewards = dataset["rewards"]
        terminals = dataset["terminals"]
        ep_rewards = []
        ep_reward = 0
        for r, t in zip(rewards, terminals):
            ep_reward += r
            if t:  # fin d'épisode
                ep_rewards.append(ep_reward)
                ep_reward = 0
    else:
        ep_rewards = episode_returns  # fallback

    n_episodes = len(episode_returns)
    episodes = np.arange(1, n_episodes + 1)

    # --- PLOT ---
    plt.figure(figsize=(15, 10))

    # 1) Episode Returns
    plt.subplot(3, 1, 1)
    plt.plot(episodes, episode_returns, label="Episode return")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("Episode Returns")
    plt.grid(True)

    # 2) Episode Lengths
    plt.subplot(3, 1, 2)
    plt.plot(episodes, episode_lengths, label="Episode length", color="orange")
    plt.xlabel("Episode")
    plt.ylabel("Length")
    plt.title("Episode Lengths")
    plt.grid(True)

    # 3) Rewards cumulés recalculés (optionnel)
    plt.subplot(3, 1, 3)
    plt.plot(episodes, ep_rewards, label="Cumulative rewards", color="green")
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Reward")
    plt.title("Episode Cumulative Rewards")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# --- Exemple d'utilisation ---

import pickle
import matplotlib.pyplot as plt
import numpy as np

def plot_full_metrics(metrics_file):
    with open(metrics_file, "rb") as f:
        dataset = pickle.load(f)

    episode_returns = dataset.get("episode_returns", [])
    episode_lengths = dataset.get("episode_lengths", [])

    # Reward moyen par étape par épisode
    rewards_per_step = []
    start = 0
    for length in episode_lengths:
        if length > 0:
            ep_rewards = dataset["rewards"][start:start+length]
            rewards_per_step.append(np.mean(ep_rewards))
        else:
            rewards_per_step.append(np.nan)
        start += length

    # Entropies et losses (par épisode)
    mean_entropy_per_episode = [np.mean(ep) if len(ep) > 0 else np.nan for ep in dataset.get("entropies", [])]
    mean_loss_per_episode = [np.mean(ep) if len(ep) > 0 else np.nan for ep in dataset.get("losses", [])]

    n_episodes = len(episode_returns)
    episodes = np.arange(1, n_episodes + 1)

    plt.figure(figsize=(15, 12))

    plt.subplot(4, 1, 1)
    plt.plot(episodes, episode_returns, label="Episode Return")
    plt.xlabel("Episode"); plt.ylabel("Return"); plt.title("Episode Return"); plt.grid(True)

    plt.subplot(4, 1, 2)
    plt.plot(episodes, episode_lengths, label="Episode Length", color="orange")
    plt.xlabel("Episode"); plt.ylabel("Length"); plt.title("Episode Length"); plt.grid(True)

    plt.subplot(4, 1, 3)
    plt.plot(episodes, rewards_per_step, label="Average reward per step", color="green")
    plt.xlabel("Episode"); plt.ylabel("Avg reward"); plt.title("Average Reward per Step"); plt.grid(True)

    plt.subplot(4, 1, 4)
    if mean_entropy_per_episode:
        plt.plot(episodes, mean_entropy_per_episode, label="Entropy", color="purple")
    if mean_loss_per_episode:
        plt.plot(episodes, mean_loss_per_episode, label="Value Loss", color="red")
    plt.xlabel("Episode"); plt.ylabel("Value"); plt.title("Entropy / Value Loss per Episode")
    plt.legend(); plt.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__": 
    # env_train = AdroitHandReachEnv(render_mode=None) 
    # env_train.debug_actuators()
    # env_train.utilis()
    #train_ppo()      # lance l'entraînement
    #visualize_trained_model()
    plot_full_metrics("C:/Users/tlamy/Sorbonne/Social robotic/Perfect_hand/logs/metrics_dataset.pkl")
    #visualize_random_actions()
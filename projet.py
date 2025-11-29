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

from shadow_hand_reach_env import AdroitHandReachEnv
from videocallback import LoggingVideoCallback  # ou le nom où tu as défini le callback
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

def train_ppo(total_timesteps=300_000,
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
        n_steps=2048,
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

    # --- SAVE MODEL ---
    model_path = os.path.join(save_dir, "ppo_shadowhand_model_v7")
    model.save(model_path)
    print(f"✔ Model saved to: {model_path}")

    # --- PLOT DES MÉTRIQUES ---
    metrics = callback.metrics
    dataset = callback.dataset

    # Random policy entropy pour référence
    action_dim = env_train.action_space.shape[0]
    random_entropy = action_dim * np.log(2.0)
    print(f"Random (uniform) policy entropy (nats) = {random_entropy:.4f} (action_dim={action_dim})")

    episode_returns = metrics.get("episode_returns", [])
    episode_lengths = metrics.get("episode_lengths", [])
    entropies = metrics.get("entropies", [])

    # Moyenne d'entropie par épisode
    mean_entropy_per_episode = []
    step_idx = 0
    for length in episode_lengths:
        ep_ent = entropies[step_idx:step_idx+length]
        mean_entropy_per_episode.append(np.mean(ep_ent) if ep_ent else np.nan)
        step_idx += length

    plt.figure(figsize=(14, 10))

    # 1) Episode returns
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
    plt.axhline(y=random_entropy, color="k", linestyle="--", linewidth=2,
                label=f"random uniform entropy = {random_entropy:.3f} nats")
    plt.xlabel("Episode")
    plt.ylabel("Entropy (nats)")
    plt.title("Policy entropy per episode (PPO) vs Random uniform")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    env_train.close()
    return model, metrics, dataset, random_entropy

import mujoco
import numpy as np
from mujoco import viewer

def visualize_trained_model():
    env_train = AdroitTrajEnv(render_mode="rgb_array", defaultsettings=True)
    # Charge le modèle entraîné
    model = PPO.load("C:/Users/tlamy/Sorbonne/Social robotic/Perfect_hand/logs/ppo_shadowhand_model_v7.zip", env=env_train)

    obs, info = env_train.reset()

    # Ouvre le viewer passif (géré par nous)
    with mujoco.viewer.launch_passive(env_train.model, env_train.data) as v:
        try:
            while True:  # boucle infinie jusqu'à Ctrl+C
                # Action du modèle
                time.sleep(0.08)
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



def plot_and_save_metrics(dataset, metrics, save_dir="plots"):
    """
    Génère les graphiques PPO et environnement individuellement et les sauvegarde.
    dataset : dict contenant 'rewards', 'actions', 'next_observations', 'terminals', etc.
    metrics : dict contenant 'episode_returns', 'episode_lengths', 'entropies',
              'value_losses', 'distances', 'terminated', 'truncated'
    save_dir : dossier où sauvegarder les plots
    """
    import os
    os.makedirs(save_dir, exist_ok=True)

    episode_returns = metrics.get("episode_returns", [])
    episode_lengths = metrics.get("episode_lengths", [])
    n_episodes = len(episode_returns)
    episodes = np.arange(1, n_episodes + 1)

    # Récompense moyenne par étape
    rewards_per_step = []
    start = 0
    for length in episode_lengths:
        if length > 0:
            ep_rewards = dataset["rewards"][start:start+length]
            rewards_per_step.append(np.mean(ep_rewards))
        else:
            rewards_per_step.append(np.nan)
        start += length

    # Fonction d'agrégation par épisode
    def aggregate_per_episode(metric_all):
        agg = []
        step_idx = 0
        for length in episode_lengths:
            if length > 0:
                ep_values = metric_all[step_idx:step_idx+length]
                agg.append(np.mean(ep_values))
            else:
                agg.append(np.nan)
            step_idx += length
        return agg

    mean_entropy_per_episode = aggregate_per_episode(metrics.get("entropies", []))
    mean_loss_per_episode = aggregate_per_episode(metrics.get("value_losses", []))
    mean_distance_per_episode = aggregate_per_episode(metrics.get("distances", []))
    terminated = np.array(metrics.get("terminated", [0]*n_episodes), dtype=int)
    truncated = np.array(metrics.get("truncated", [0]*n_episodes), dtype=int)

    plots = []

    # --- Plot 1 : Episode Return ---
    fig1, ax1 = plt.subplots(figsize=(10,5))
    ax1.plot(episodes, episode_returns, label="Episode Return", color="blue")
    ax1.set_xlabel("Episode"); ax1.set_ylabel("Return"); ax1.set_title("Episode Return")
    ax1.grid(True)
    fig1.savefig(os.path.join(save_dir, "episode_return.png"))
    plots.append(fig1)

    # --- Plot 2 : Episode Length + Termination ---
    fig2, ax2 = plt.subplots(figsize=(10,5))
    ax2.plot(episodes, episode_lengths, label="Episode Length", color="orange")
    ax2.plot(episodes, terminated, label="Terminated", color="red", linestyle="--")
    ax2.plot(episodes, truncated, label="Truncated", color="black", linestyle="--")
    print(terminated)
    print(truncated)
    ax2.set_xlabel("Episode"); ax2.set_ylabel("Value"); ax2.set_title("Episode Length & Termination")
    ax2.legend(); ax2.grid(True)
    fig2.savefig(os.path.join(save_dir, "length_termination.png"))
    plots.append(fig2)

    # --- Plot 3 : Average Reward per Step ---
    fig3, ax3 = plt.subplots(figsize=(10,5))
    ax3.plot(episodes, rewards_per_step, label="Avg Reward per Step", color="green")
    ax3.set_xlabel("Episode"); ax3.set_ylabel("Avg Reward"); ax3.set_title("Average Reward per Step")
    ax3.grid(True)
    fig3.savefig(os.path.join(save_dir, "avg_reward_per_step.png"))
    plots.append(fig3)

    # --- Plot 4 : Entropy / Value Loss ---
    fig4, ax4 = plt.subplots(figsize=(10,5))
    if mean_entropy_per_episode:
        ax4.plot(episodes, mean_entropy_per_episode, label="Entropy", color="purple")
    if mean_loss_per_episode:
        ax4.plot(episodes, mean_loss_per_episode, label="Value Loss", color="red")
    ax4.set_xlabel("Episode"); ax4.set_ylabel("Value"); ax4.set_title("Entropy / Value Loss per Episode")
    ax4.legend(); ax4.grid(True)
    fig4.savefig(os.path.join(save_dir, "entropy_value_loss.png"))
    plots.append(fig4)

    # --- Plot 5 : Finger Distance ---
    fig5, ax5 = plt.subplots(figsize=(10,5))
    if mean_distance_per_episode:
        ax5.plot(episodes, mean_distance_per_episode, label="Finger Distance", color="blue")
    ax5.set_xlabel("Episode"); ax5.set_ylabel("Distance"); ax5.set_title("Finger Distance per Episode")
    ax5.legend(); ax5.grid(True)
    fig5.savefig(os.path.join(save_dir, "finger_distance.png"))
    plots.append(fig5)
    
    fig6, ax6 = plt.subplots(figsize=(10,5))
    rewards = dataset["rewards"]
    plt.plot(np.arange(len(rewards)), rewards)
    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.title("Reward per Step")
    ax6.set_xlabel("Episode"); ax6.set_ylabel("Distance"); ax6.set_title("Finger Distance per Episode")
    ax6.legend(); ax6.grid(True)
    fig6.savefig(os.path.join(save_dir, "reward per episode.png"))
    plots.append(fig6)

    print(f"✔ Plots saved in {save_dir}")
    return plots

def plot_full_metrics(dataset, metrics):
    """
    Plot complet des métriques PPO et de l'environnement.
    dataset : dict contenant 'rewards', 'actions', 'next_observations', 'terminals', etc.
    metrics : dict contenant 'episode_returns', 'episode_lengths', 'entropies',
              'value_losses', 'distances', 'terminated', 'truncated'
    """

    # --------------------------
    # Episodes & longueurs
    # --------------------------
    episode_returns = metrics.get("episode_returns", [])
    episode_lengths = metrics.get("episode_lengths", [])
    n_episodes = len(episode_returns)
    episodes = np.arange(1, n_episodes + 1)

    # --------------------------
    # Récompense moyenne par étape par épisode
    # --------------------------
    rewards_per_step = []
    start = 0
    for length in episode_lengths:
        if length > 0:
            ep_rewards = dataset["rewards"][start:start+length]
            rewards_per_step.append(np.mean(ep_rewards))
        else:
            rewards_per_step.append(np.nan)
        start += length

    # --------------------------
    # Agrégation des métriques par épisode
    # --------------------------
    def aggregate_per_episode(metric_all):
        """Transforme une métrique par step en moyenne par épisode"""
        agg = []
        step_idx = 0
        for length in episode_lengths:
            if length > 0:
                ep_values = metric_all[step_idx:step_idx+length]
                agg.append(np.mean(ep_values))
            else:
                agg.append(np.nan)
            step_idx += length
        return agg

    mean_entropy_per_episode = aggregate_per_episode(metrics.get("entropies", []))
    mean_loss_per_episode = aggregate_per_episode(metrics.get("value_losses", []))
    mean_distance_per_episode = aggregate_per_episode(metrics.get("distances", []))

    # --------------------------
    # Tracé des graphiques
    # --------------------------
    plt.figure(figsize=(16, 14))

    # 1) Episode Return
    plt.subplot(6, 1, 1)
    plt.plot(episodes, episode_returns, label="Episode Return", color="blue")
    plt.xlabel("Episode"); plt.ylabel("Return"); plt.title("Episode Return"); plt.grid(True)

    # 2) Episode Length
    plt.subplot(6, 1, 2)
    plt.plot(episodes, episode_lengths, label="Episode Length", color="orange")
    plt.xlabel("Episode"); plt.ylabel("Length"); plt.title("Episode Length"); plt.grid(True)

    # 3) Average Reward per Step
    plt.subplot(6, 1, 3)
    plt.plot(episodes, rewards_per_step, label="Avg Reward per Step", color="green")
    plt.xlabel("Episode"); plt.ylabel("Avg Reward"); plt.title("Average Reward per Step"); plt.grid(True)

    # 4) Entropy / Value Loss
    plt.subplot(6, 1, 4)
    if mean_entropy_per_episode:
        plt.plot(episodes, mean_entropy_per_episode, label="Entropy", color="purple")
    if mean_loss_per_episode:
        plt.plot(episodes, mean_loss_per_episode, label="Value Loss", color="red")
    plt.xlabel("Episode"); plt.ylabel("Value"); plt.title("Entropy / Value Loss per Episode")
    plt.legend(); plt.grid(True)

    # 5) Distance & Termination
    plt.subplot(6, 1, 5)
   
    terminated = np.array(metrics.get("terminated", [0]*n_episodes), dtype=int)
    truncated = np.array(metrics.get("truncated", [0]*n_episodes), dtype=int)

    plt.plot(episodes, terminated, label="Terminated", color="red", linestyle="--")
    plt.plot(episodes, truncated, label="Truncated", color="black", linestyle="--")
    plt.xlabel("Episode"); plt.ylabel("Termination"); plt.title( "Episode Termination")
    plt.legend(); plt.grid(True)
    plt.subplot(6, 1, 6)
    if mean_distance_per_episode:
        plt.plot(episodes, mean_distance_per_episode, label="Finger Distance", color="blue")
        plt.xlabel("Episode"); plt.ylabel("Finger Distance"); plt.title( "Episode Finger Distance")
        plt.legend(); plt.grid(True)
    plt.tight_layout()
    plt.show()
if __name__ == "__main__": 
    train_ppo()      # lance l'entraînement
    #visualize_trained_model()
    # with open("C:/Users/tlamy/Sorbonne/Social robotic/Perfect_hand/logs/metrics/metrics3.pkl", "rb") as f:
    #     metrics = pickle.load(f)
    # with open("C:/Users/tlamy/Sorbonne/Social robotic/Perfect_hand/logs/dataset/dataset_episodessss_97.pkl", "rb") as f:
    #     dataset = pickle.load(f)
    # plot_and_save_metrics(dataset,metrics)
    # plot_full_metrics(metrics=metrics,dataset=dataset)
    

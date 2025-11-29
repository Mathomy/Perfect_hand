
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
from videocallback import LoggingVideoCallback  
def train_ppo(total_timesteps=100_000,
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
    model_path = os.path.join(save_dir, "ppo_shadowhand_model_v9")
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



def visualize_trained_model():
    env_train = AdroitTrajEnv(render_mode="rgb_array", defaultsettings=True)
    # Charge le modèle entraîné
    model = PPO.load("C:/Users/tlamy/Sorbonne/Social robotic/Perfect_hand/logs/ppo_shadowhand_model_v9.zip", env=env_train)

    obs, info = env_train.reset()

    # Ouvre le viewer passif (géré par nous)
    with mujoco.viewer.launch_passive(env_train.model, env_train.data) as v:
        try:
            while True:  # boucle infinie jusqu'à Ctrl+C
                # Action du modèle
                time.sleep(0.2)
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env_train.step(action)

                # Debug simple si tu veux voir ce qui se passe
                print("dist:", info.get("dist_fingers"), "reward:", reward)

                # Afficher la frame
                v.sync()
                print(terminated,truncated)
                # Si épisode fini → on reset mais on ne ferme PAS la fenêtre
                if terminated or truncated:
                    time.sleep(5)
                    obs, info = env_train.reset()

        except KeyboardInterrupt:
            print("Visualisation interrompue par l'utilisateur (Ctrl+C).")

    env.close()


def plot_and_save_metrics(dataset, metrics, save_dir="plots_short"):
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
    n_episodes = 2000
    episodes = np.arange(1, n_episodes + 1)

    # Récompense moyenne par étape
    rewards_per_step = []
    start = 0
    for length in episode_lengths[::episodes] :
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
        for length in episode_lengths[::episodes]:
            if length > 0:
                ep_values = metric_all[step_idx:step_idx+length]
                agg.append(np.mean(ep_values))
            else:
                agg.append(np.nan)
            step_idx += length
        return agg
    def aggregate_min_per_episode(metric_all):
        """Transforme une métrique par step en minimum par épisode"""
        agg = []
        step_idx = 0
        for length in episode_lengths[::episodes]:
            ep_values = metric_all[step_idx:step_idx+length]
            if len(ep_values) > 0:
                print(ep_values)
                agg.append(np.min(ep_values))
            else:
                agg.append(np.nan)  # ou None si tu préfères
            step_idx += length
        return agg

    mean_entropy_per_episode = aggregate_per_episode(metrics.get("entropies", []))
    mean_loss_per_episode = aggregate_per_episode(metrics.get("value_losses", []))
    min_distance_per_episode = aggregate_min_per_episode(metrics.get("distances_finger", []))
    min_target_disitance_per_episode = aggregate_min_per_episode(metrics.get("distances_target", []))
    terminated = np.array(metrics.get("terminated", [0]*n_episodes), dtype=int)
    truncated = np.array(metrics.get("truncated", [0]*n_episodes), dtype=int)

    plots = []

    # --- Plot 1 : Episode Return ---
    fig1, ax1 = plt.subplots(figsize=(15,5))
    ax1.plot(episodes, episode_returns[::2000], label="Episode Return", color="blue")
    ax1.set_xlabel("Episode"); ax1.set_ylabel("Return"); ax1.set_title("Episode Return")
    ax1.grid(True)
    fig1.savefig(os.path.join(save_dir, "episode_return.png"))
    plots.append(fig1)

    # --- Plot 2 : Episode Length ---
    fig2, ax2 = plt.subplots(figsize=(15,5))
    ax2.plot(episodes, episode_lengths, label="Episode Length", color="orange")
    ax2.legend(); ax2.grid(True)
    fig2.savefig(os.path.join(save_dir, "length.png"))
    plots.append(fig2)

    # --- Plot 3 : Average Reward per Step ---
    fig3, ax3 = plt.subplots(figsize=(15,5))
    ax3.scatter(episodes, rewards_per_step, label="Avg Reward per Step", color="green",s=0.1)
    ax3.set_xlabel("Episode"); ax3.set_ylabel("Avg Reward"); ax3.set_title("Average Reward per Step")
    ax3.grid(True)
    fig3.savefig(os.path.join(save_dir, "avg_reward_per_step.png"))
    plots.append(fig3)

    # --- Plot 4 : Entropy / Value Loss ---
    fig4, ax4 = plt.subplots(figsize=(15,5))
    if mean_entropy_per_episode:
        ax4.plot(episodes, mean_entropy_per_episode, label="Entropy", color="purple")
    ax4.set_xlabel("Episode"); ax4.set_ylabel("Value"); ax4.set_title("Entropy per Episode")
    ax4.legend(); ax4.grid(True)
    fig4.savefig(os.path.join(save_dir, "entropy.png"))
    plots.append(fig4)

    # --- Plot 5 : Finger Distance ---
    fig5, ax5 = plt.subplots(figsize=(15,5))
    if min_distance_per_episode:
        ax5.scatter(episodes, min_distance_per_episode, label="Finger Distance", color="blue",s=0.1)
        # ax5.scatter(episodes, min_target_disitance_per_episode, label="target Distance", color="red",s=0.1)
    ax5.set_xlabel("Episode"); ax5.set_ylabel("Distance"); ax5.set_title("Finger Distance per Episode")
    ax5.legend(); ax5.grid(True)
    fig5.savefig(os.path.join(save_dir, "finger_distance.png"))
    plots.append(fig5)
    

    print(f"✔ Plots saved in {save_dir}")
    plt.show()
    return plots

if __name__ == "__main__": 
    # train_ppo()      # lance l'entraînement
    # visualize_trained_model()
    with open("C:/Users/tlamy/Sorbonne/Social robotic/Perfect_hand/logs/metrics/metrics4.pkl", "rb") as f:
        metrics = pickle.load(f)
    with open("C:/Users/tlamy/Sorbonne/Social robotic/Perfect_hand/logs/dataset/dataset_final.pkl", "rb") as f:
        dataset = pickle.load(f)
    plot_and_save_metrics(dataset,metrics)
    

import os
import time
import pickle
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import mujoco

# Import your modules
from adroit_env_datafocused import AdroitTrajEnv
from videocallback import LoggingVideoCallback


def train_ppo():
    """
    Train PPO while collecting a proper dataset for PEBBLE/D4RL.
    Uses neutral position start for all episodes.
    """
    os.makedirs("logs", exist_ok=True)
    
    env_train = DummyVecEnv([lambda:AdroitTrajEnv(render_mode="rgb_array", defaultsettings=True,camera_name="side_view")])

    #callback for dataset collection
    videocall = LoggingVideoCallback(
        env=env_train,
        save_dir="logs",
        video_freq=30, 
        verbose=1,
        frame_skip=3
    )

    # Create PPO model with improved hyperparameters
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
        tensorboard_log="./ppo_shadowhand/"
    )

    print("Starting training with neutral position start...")
    model.learn(total_timesteps=300000, callback=videocall)

    model.save("ppo_shadowhand")
    env_train.close()
    
    print("✓ Entraînement terminé et modèle sauvegardé.")
    print(f"✓ Dataset saved with {len(videocall.dataset['episode_returns'])} episodes")




def visualize_trained_model():
    env_train = AdroitTrajEnv(render_mode="rgb_array", defaultsettings=True)
    # Charge le modèle entraîné
    model = PPO.load("ppo_shadowhand.zip", env=env_train)

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

    env_train.close()




def evaluate_model(): #Evaluate the trained model and collect statistics
    """
    Evaluate the trained model and collect statistics.
    """
    env = AdroitTrajEnv(render_mode=None, defaultsettings=True)
    model = PPO.load("ppo_shadowhand.zip", env=env)

    num_episodes = 100
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    best_pinch_distances = []

    print("Evaluating model over 100 episodes...")
    
    for ep in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_length = 0
        best_pinch = float('inf')
        
        for _ in range(1000):  # Max 1000 steps per episode
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_length += 1
            best_pinch = min(best_pinch, info['dist_fingers'])
            
            if terminated or truncated:
                if info['dist_fingers'] < 0.015:
                    success_count += 1
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        best_pinch_distances.append(best_pinch)
        
        if (ep + 1) % 20 == 0:
            print(f"Episode {ep + 1}/{num_episodes} complete")

    print("\n=== Evaluation Results ===")
    print(f"Success rate: {success_count / num_episodes * 100:.1f}%")
    print(f"Average reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Average episode length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"Average best pinch distance: {np.mean(best_pinch_distances):.4f} ± {np.std(best_pinch_distances):.4f}")
    
    env.close()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "train":
            train_ppo()
        elif command == "visualize":
            visualize_trained_model()
        elif command == "evaluate":
            evaluate_model()
        else:
            print("Unknown command.")
            print("Available commands: train, visualize, evaluate")
    else:
        # Default: run training
        print("Starting training (use 'python training_ppo.py <command>' for other options) (or 'mjpython training_ppo.py <command>' for macOS users.)")
        print("Available commands: train, analyze, convert, visualize, evaluate")
        train_ppo()

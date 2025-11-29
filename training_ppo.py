import os
import pickle
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import mujoco

# Import your modules
from adroit_env_datafocused import AdroitTrajEnv
from videocallback import VideoCallback


def train_ppo():
    """
    Train PPO while collecting a proper dataset for PEBBLE/D4RL.
    Uses neutral position start for all episodes.
    """
    os.makedirs("logs", exist_ok=True)
    
    env_train = DummyVecEnv([lambda:AdroitTrajEnv(render_mode="rgb_array", defaultsettings=True)])

    #callback for dataset collection
    videocall = VideoCallback(
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
    model.learn(total_timesteps=100000, callback=videocall)

    model.save("ppo_shadowhand")
    env_train.close()
    
    print("✓ Entraînement terminé et modèle sauvegardé.")
    print(f"✓ Dataset saved with {len(videocall.dataset['episode_returns'])} episodes")




def visualize_trained_model():
    """
    Visualiser le modèle entraîné en train de pincer.
    La fenêtre reste ouverte jusqu'à ce que l'utilisateur la ferme.
    """
    env = AdroitTrajEnv(render_mode=None, defaultsettings=True)
    
    # Charger le modèle entraîné
    model = PPO.load("ppo_shadowhand.zip", env=env)

    obs, info = env.reset()

    
    with mujoco.viewer.launch_passive(env.model, env.data) as v:
        step_count = 0
        episode_count = 0

        try:
            while v.is_running():  # Tant que la fenêtre n'est pas fermée
                # Obtenir l'action du modèle
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)

                # Afficher les infos de debug tous les 10 pas
                if step_count % 10 == 0:
                    print(f"Step {step_count}: "
                          f"Pinch dist: {info['dist_fingers']:.4f}, "
                          f"Target dist: {info['dist_to_target']:.4f}, "
                          f"Straightness: {info.get('straightness', 0.0):.3f}, "
                          f"Reward: {reward:.2f}")

                # Synchroniser l'affichage
                v.sync()
                step_count += 1

                # Reset à la fin de l'épisode, garder la fenêtre ouverte
                if terminated or truncated:
                    episode_count += 1
                    print(f"\n=== Épisode {episode_count} terminé ===")
                    print(f"Distance finale de pincement: {info['dist_fingers']:.4f}\n")
                    obs, info = env.reset()
                    step_count = 0

        except KeyboardInterrupt:
            print("Visualisation interrompue par l'utilisateur (Ctrl+C).")
        finally:
            env.close()




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

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import cv2
from typing import List, Dict
import gym
import gymnasium as gym
import mujoco
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO
from adroit_env_datafocused_temp import AdroitTrajEnv



class RewardModel(nn.Module):
    """Neural network that learns to predict rewards from state-action pairs."""
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = [256, 256]):
        super().__init__()
        layers = []
        input_dim = state_dim + action_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(hidden_dim))
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.network(x)


class PreferenceDataset:
    """
    Converts and stores preference data in PbRL format.
    """
    def __init__(self, max_size: int = 100000):
        self.preferences = []
        self.max_size = max_size
    
    def add_preference(self, pref_entry: Dict):
        if len(self.preferences) >= self.max_size:
            self.preferences.pop(0)
        
        converted = {
            'segment0': pref_entry['sigma_0_traj'],
            'segment1': pref_entry['sigma_1_traj'],
            'label': pref_entry['preference']
        }
        self.preferences.append(converted)
    
    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self.preferences, f)
    
    def load(self, filepath):
        with open(filepath, 'rb') as f:
            self.preferences = pickle.load(f)
    
    def __len__(self):
        return len(self.preferences)


class PreferenceRewardLearner:
    """Learns reward function from preferences using Bradley-Terry model."""
    def __init__(self, state_dim: int, action_dim: int,
                 lr: float = 3e-4, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.reward_model = RewardModel(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.reward_model.parameters(), lr=lr)
    
    def compute_trajectory_reward(self, segment: List[Dict]) -> torch.Tensor:
        steps = [{'obs': obs, 'action': act} 
        for obs, act in zip(segment['observations'], segment['actions'])]

        states = torch.FloatTensor([step['obs'] for step in steps]).to(self.device)
        actions = torch.FloatTensor([step['action'] for step in steps]).to(self.device)
        rewards = self.reward_model(states, actions)

        return rewards.sum()
    
    def compute_preference_probability(self, segment0: List[Dict], segment1: List[Dict]) -> torch.Tensor:
        r0_sum = self.compute_trajectory_reward(segment0)
        r1_sum = self.compute_trajectory_reward(segment1)
        logits = torch.stack([r0_sum, r1_sum])
        probs = torch.softmax(logits, dim=0)
        return probs[0]
    
    def train_step(self, preference_batch: List[Dict]) -> float:
        self.optimizer.zero_grad()
        total_loss = 0.0
        for pref in preference_batch:
            segment0 = pref['segment0']
            segment1 = pref['segment1']
            label = pref['label']
            prob_0_better = self.compute_preference_probability(segment0, segment1)
            if label == 0:
                loss = -torch.log(prob_0_better + 1e-8)
            elif label == 1:
                loss = -torch.log(1 - prob_0_better + 1e-8)
            else:
                loss = -(0.5 * torch.log(prob_0_better + 1e-8) +
                         0.5 * torch.log(1 - prob_0_better + 1e-8))
            total_loss += loss
        avg_loss = total_loss / len(preference_batch)
        avg_loss.backward()
        self.optimizer.step()
        return avg_loss.item()
    
    def train(self, dataset: PreferenceDataset, epochs: int = 100, batch_size: int = 32):
        losses = []
        for epoch in range(epochs):
            indices = np.random.permutation(len(dataset))
            epoch_losses = []
            for i in range(0, len(dataset), batch_size):
                batch_indices = indices[i:i+batch_size]
                batch = [dataset.preferences[idx] for idx in batch_indices]
                loss = self.train_step(batch)
                epoch_losses.append(loss)
            avg_loss = np.mean(epoch_losses)
            losses.append(avg_loss)
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        return losses
    
    def predict_reward(self, state: np.ndarray, action: np.ndarray) -> float:
        self.reward_model.eval()
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action_t = torch.FloatTensor(action).unsqueeze(0).to(self.device)
            reward = self.reward_model(state_t, action_t)
        return reward.item()
    
    def save(self, filepath: str):
        torch.save({
            'model_state_dict': self.reward_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filepath)
    
    def load(self, filepath: str):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.reward_model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


# Load your existing preference dataset
def load_existing_preference_data(filepath: str) -> PreferenceDataset:
    """
    Load your custom preference dataset and convert it to PbRL format.
    """
    with open(filepath, 'rb') as f:
        raw_data = pickle.load(f)
    
    dataset = PreferenceDataset()
    for entry in raw_data:
        dataset.add_preference(entry)
    
    print(f"Loaded {len(dataset)} preferences from existing file.")
    return dataset


# PbRL Wrapper
class PbRLWrapper(gym.Wrapper):
    """Wrap environment to replace rewards with learned reward."""
    def __init__(self, env, reward_learner: PreferenceRewardLearner):
        super().__init__(env)
        self.reward_learner = reward_learner
    
    def step(self, action):
        obs, original_reward, terminated, truncated, info = self.env.step(action)
        learned_reward = self.reward_learner.predict_reward(obs, action)
        info['original_reward'] = original_reward
        info['learned_reward'] = learned_reward
        return obs, learned_reward, terminated, truncated, info



def train_with_pbrl():
    print("Loading existing preference dataset...")
    pref_dataset = load_existing_preference_data("logs/dataset/sigma_data.pkl")
    
    # --- Reward model training ---
    env = AdroitTrajEnv(defaultsettings=True)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    reward_learner = PreferenceRewardLearner(state_dim, action_dim)
    reward_learner.train(pref_dataset, epochs=100, batch_size=32)
    reward_learner.save("reward_model.pth")
    print("Reward model trained and saved.")
    
    # --- PPO with learned rewards ---
    wrapped_env = PbRLWrapper(env, reward_learner)
    vec_env = DummyVecEnv([lambda: wrapped_env])
    model = PPO("MlpPolicy", vec_env, verbose=1)
    model.learn(total_timesteps=250_000)
    model.save("ppo_pbrl")
    print("PbRL training complete.")


def visualize_trained_model():
    """
    Visualiser le modèle entraîné en train de pincer.
    La fenêtre reste ouverte jusqu'à ce que l'utilisateur la ferme.
    """
    env = AdroitTrajEnv(render_mode=None, defaultsettings=True)
    
    # Charger le modèle entraîné
    model = PPO.load("ppo_pbrl.zip", env=env)

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

def evaluate_model(): 
    """
    Evaluate the trained model and collect statistics.
    """
    env = AdroitTrajEnv(render_mode=None, defaultsettings=True)
    model = PPO.load("ppo_pbrl.zip", env=env)

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
    train_with_pbrl()
    visualize_trained_model()
    evaluate_model()
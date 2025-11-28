import os
import pickle
import imageio
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class LoggingVideoCallback(BaseCallback):
    """
    Callback combinant :
    - Enregistrement des vidéos tous les N épisodes ou pour succès
    - Construction d’un dataset de trajectoires
    - Enregistrement des métriques d’entraînement : reward moyen, entropie, value loss, longueur épisode
    """
    def __init__(self, env, save_dir="logs", video_freq=10, save_success_videos=True, verbose=1, frame_skip=1):
        super().__init__(verbose)
        self.env = env
        self.save_dir = save_dir
        self.video_freq = video_freq
        self.save_success_videos = save_success_videos
        self.frame_skip = frame_skip

        self.episode_count = 0
        self.episode_step_count = 0
        self.frames = []

        # Trajectoire en cours
        self.current_episode = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'next_observations': [],
            'terminals': [],
            'infos': []
        }

        # Dataset complet
        self.dataset = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'next_observations': [],
            'terminals': [],
            'episode_returns': [],
            'episode_lengths': []
        }

        # Mapping vidéo / trajectoire
        self.video_trajectory_map = []

        # --- MÉTRIQUES POUR TRACÉ ---
        self.metrics = {
            'episode_returns': [],
            'episode_lengths': [],
            'entropies': [],
            'value_losses': []
        }

        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(f"{save_dir}/videos", exist_ok=True)
        os.makedirs(f"{save_dir}/trajectories", exist_ok=True)
        os.makedirs(f"{save_dir}/dataset", exist_ok=True)
        os.makedirs(f"{save_dir}/successful_episodes", exist_ok=True)

    def _on_rollout_start(self):
        """Initialiser la capture vidéo."""
        try:
            frame = self.env.envs[0].get_frame()
            self.frames = [frame]
        except:
            self.frames = []

    def _on_step(self) -> bool:
        env_idx = 0
        self.episode_step_count += 1

        # Capture frames
        if self.episode_step_count % self.frame_skip == 0:
            if hasattr(self.env.envs[env_idx], "get_frame"):
                self.frames.append(self.env.envs[env_idx].get_frame())

        # Récupérer transition
        obs = self.locals["new_obs"][env_idx].copy()
        action = self.locals["actions"][env_idx].copy()
        reward = self.locals["rewards"][env_idx]
        info = self.locals["infos"][env_idx]

        prev_obs = self.current_episode['observations'][-1] if self.current_episode['observations'] else obs

        # Stocker transition
        self.current_episode['observations'].append(prev_obs)
        self.current_episode['actions'].append(action)
        self.current_episode['rewards'].append(reward)
        self.current_episode['next_observations'].append(obs)
        self.current_episode['infos'].append(info)

        done = self.locals["dones"][env_idx]

        # --- MÉTRIQUES PPO ---
        # Entropie
        obs_tensor, _ = self.model.policy.obs_to_tensor(np.expand_dims(obs, axis=0))
        dist = self.model.policy.get_distribution(obs_tensor)
        self.metrics['entropies'].append(dist.entropy().mean().item())

        # Value loss si disponible
        if hasattr(self.model, "value_loss"):
            self.metrics['value_losses'].append(self.model.value_loss)

        if done:
            self.episode_count += 1
            self.current_episode['terminals'] = [False]*(len(self.current_episode['actions'])-1) + [True]

            ep_return = sum(self.current_episode['rewards'])
            ep_length = len(self.current_episode['rewards'])
            is_success = info.get('success', False) or info.get('dist_fingers', 1.0) < 0.015

            # Ajout aux métriques
            self.metrics['episode_returns'].append(ep_return)
            self.metrics['episode_lengths'].append(ep_length)

            # Sauvegarder dataset et trajectoire
            self.dataset['observations'].extend(self.current_episode['observations'])
            self.dataset['actions'].extend(self.current_episode['actions'])
            self.dataset['rewards'].extend(self.current_episode['rewards'])
            self.dataset['next_observations'].extend(self.current_episode['next_observations'])
            self.dataset['terminals'].extend(self.current_episode['terminals'])
            self.dataset['episode_returns'].append(ep_return)
            self.dataset['episode_lengths'].append(ep_length)
           

            traj_file = f"{self.save_dir}/trajectories/episode_{self.episode_count:04d}.pkl"
            with open(traj_file, "wb") as f:
                pickle.dump(self.current_episode, f)

            # --- Vidéo si conditions ---
            save_video = False
            reason = ""
            if self.episode_count % self.video_freq == 0:
                save_video = True
                reason = "periodic"
            if self.save_success_videos and is_success:
                save_video = True
                reason = "success" if reason == "" else "periodic+success"

            if save_video and self.frames:
                video_file = f"{self.save_dir}/videos/episode_{self.episode_count:04d}.mp4"
                imageio.mimsave(video_file, self.frames, fps=30)
                self.video_trajectory_map.append({
                    'episode': self.episode_count,
                    'video': video_file,
                    'trajectory': traj_file,
                    'return': ep_return,
                    'length': ep_length,
                    'success': is_success,
                    'reason': reason
                })

            # Réinitialiser épisode
            self.frames = []
            self.current_episode = {
                'observations': [],
                'actions': [],
                'rewards': [],
                'next_observations': [],
                'terminals': [],
                'infos': []
            }
            self.episode_step_count = 0

        return True

    def _save_dataset(self):
        """Sauvegarder le dataset complet."""
        dataset_np = {k: np.array(v) for k, v in self.dataset.items()}
        dataset_file = f"{self.save_dir}/dataset/dataset_episodes_{self.episode_count}.pkl"
        with open(dataset_file, "wb") as f:
            pickle.dump(dataset_np, f)
        if self.verbose > 0:
            print(f"Dataset sauvegardé: {len(self.dataset['episode_returns'])} épisodes, {len(self.dataset['observations'])} transitions")

    def _on_training_end(self):
        self._save_dataset()
        mapping_file = f"{self.save_dir}/video_trajectory_mapping.pkl"
        with open(mapping_file, "wb") as f:
            pickle.dump(self.video_trajectory_map, f)
        print(f"\n=== Training terminé ===")
        print(f"Dataset final: {len(self.dataset['episode_returns'])} épisodes")
        print(f"Vidéos sauvegardées: {len(self.video_trajectory_map)}")
        print(f"Épisodes réussis: {sum(1 for m in self.video_trajectory_map if m['success'])}")
        print(f"Mapping vidéo-trajectoire sauvegardé: {mapping_file}")

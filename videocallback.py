import os
import pickle
import imageio
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class VideoCallback(BaseCallback):
    """
    Callback amélioré pour créer des datasets et des vidéos sélectives.
    Sauvegarde des vidéos tous les N épisodes ET pour chaque succès.
    """
    def __init__(self, env, save_dir="logs", video_freq=10, save_success_videos=True, verbose=1, frame_skip=1):
        super().__init__(verbose)
        self.env = env
        self.save_dir = save_dir
        self.video_freq = video_freq  # Sauvegarder vidéo tous les N épisodes
        self.save_success_videos = save_success_videos  # Sauvegarder les vidéos qui atteignent le goal
        self.frame_skip = frame_skip

        self.episode_count = 0
        self.episode_step_count = 0
        self.frames = []
        
        # Stockage de trajectoire amélioré
        self.current_episode = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'next_observations': [],
            'terminals': [],
            'infos': []
        }
        
        # Stockage du dataset (tous les épisodes)
        self.dataset = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'next_observations': [],
            'terminals': [],
            'episode_returns': [],
            'episode_lengths': []
        }
        
        # Mapping entre vidéos et trajectoires
        self.video_trajectory_map = []

        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(f"{save_dir}/videos", exist_ok=True)
        os.makedirs(f"{save_dir}/trajectories", exist_ok=True)
        os.makedirs(f"{save_dir}/dataset", exist_ok=True)
        os.makedirs(f"{save_dir}/successful_episodes", exist_ok=True)

    def _on_rollout_start(self):
        """Called at the beginning of each rollout, right after reset()."""
        try:
            frame = self.env.envs[0].get_frame()
            self.frames = [frame]
        except:
            self.frames = []  


    def _on_step(self) -> bool:
        """Appelé à chaque étape de l'environnement."""
        env_idx = 0

        # Collecter la frame
        self.episode_step_count += 1
        if self.episode_step_count % self.frame_skip == 0:
            if hasattr(self.env.envs[env_idx], "get_frame"):
                self.frames.append(self.env.envs[env_idx].get_frame())

        # Stocker les données de transition
        obs = self.locals["new_obs"][env_idx].copy()
        action = self.locals["actions"][env_idx].copy()
        reward = self.locals["rewards"][env_idx]
        info = self.locals["infos"][env_idx]
        
        # Obtenir l'observation précédente (obs actuelle avant cette étape)
        if len(self.current_episode['observations']) > 0:
            prev_obs = self.current_episode['observations'][-1]
        else:
            # Première étape de l'épisode - utiliser l'observation initiale
            prev_obs = obs

        # Stocker la transition
        self.current_episode['observations'].append(prev_obs)
        self.current_episode['actions'].append(action)
        self.current_episode['rewards'].append(reward)
        self.current_episode['next_observations'].append(obs)
        self.current_episode['infos'].append(info)

        # Vérifier la terminaison de l'épisode
        done = self.locals["dones"][env_idx]

        if done:
            self.episode_count += 1
            
            # Marquer la dernière transition comme terminale
            self.current_episode['terminals'] = [False] * (len(self.current_episode['actions']) - 1) + [True]
            
            # Calculer les statistiques de l'épisode
            episode_return = sum(self.current_episode['rewards'])
            episode_length = len(self.current_episode['rewards'])
            
            # Vérifier si l'épisode a réussi (atteint le goal)
            is_success = info.get('success', False) or info.get('dist_fingers', 1.0) < 0.015
            
            # Ajouter au dataset
            self.dataset['observations'].extend(self.current_episode['observations'])
            self.dataset['actions'].extend(self.current_episode['actions'])
            self.dataset['rewards'].extend(self.current_episode['rewards'])
            self.dataset['next_observations'].extend(self.current_episode['next_observations'])
            self.dataset['terminals'].extend(self.current_episode['terminals'])
            self.dataset['episode_returns'].append(episode_return)
            self.dataset['episode_lengths'].append(episode_length)

            # Sauvegarder la trajectoire individuelle (toujours)
            traj_file = f"{self.save_dir}/trajectories/episode_{self.episode_count:04d}.pkl"
            with open(traj_file, "wb") as f:
                pickle.dump(self.current_episode, f)

            # Décider si on doit sauvegarder la vidéo
            should_save_video = False
            video_reason = ""
            
            # Use episode return instead of list
            episode_return = sum(self.current_episode['rewards'])
            if episode_return > -100:
                #  Sauvegarde chaque N épisodes
                if self.episode_count % self.video_freq == 0:
                    should_save_video = True
                    video_reason = "periodic"
                
                # Sauvegarde si succès
                if self.save_success_videos and is_success:
                    should_save_video = True
                    video_reason = "success" if video_reason == "" else "periodic+success"
                
                # Sauvegarder la vidéo si les critères sont remplis
                if should_save_video and len(self.frames) > 0:
                    video_file = f"{self.save_dir}/videos/episode_{self.episode_count:04d}.mp4"
                    imageio.mimsave(video_file, self.frames, fps=30)
                    
                    # Sauvegarder le mapping entre vidéo et trajectoire
                    self.video_trajectory_map.append({
                        'episode_num': self.episode_count,
                        'video_path': video_file,
                        'trajectory_path': traj_file,
                        'return': episode_return,
                        'length': episode_length,
                        'success': is_success,
                        'reason': video_reason
                    })
                    
                    # Si réussi, sauvegarder aussi une copie dans successful_episodes
                    if is_success:
                        success_video = f"{self.save_dir}/successful_episodes/episode_{self.episode_count:04d}.mp4"
                        success_traj = f"{self.save_dir}/successful_episodes/trajectory_{self.episode_count:04d}.pkl"
                        imageio.mimsave(success_video, self.frames, fps=5)
                        with open(success_traj, "wb") as f:
                            pickle.dump(self.current_episode, f)
                    
                    if self.verbose > 0:
                        success_str = " ✓ SUCCESS" if is_success else ""
                        print(f"[Episode {self.episode_count}] Return: {episode_return:.2f}, "
                            f"Length: {episode_length}, Reason: {video_reason}{success_str}, "
                            f"Saved video ({len(self.frames)} frames)")
                        
            elif self.verbose > 0:
                # Afficher les stats même sans vidéo
                success_str = " ✓ SUCCESS" if is_success else ""
                print(f"[Episode {self.episode_count}] Return: {episode_return:.2f}, "
                      f"Length: {episode_length}{success_str}")

            # Sauvegarder le dataset complet périodiquement
            if self.episode_count % 50 == 0:
                self._save_dataset()

            # Réinitialiser les buffers
            self.frames = []
            self.current_episode = {
                'observations': [],
                'actions': [],
                'rewards': [],
                'next_observations': [],
                'terminals': [],
                'infos': []
            }

            first_frame = self.env.envs[0].get_frame()
            self.frames.append(first_frame)
            self.episode_step_count = 0

        return True

    def _save_dataset(self):
        """Sauvegarder le dataset complet au format compatible D4RL."""
        # Convertir les listes en arrays numpy
        dataset_np = {
            'observations': np.array(self.dataset['observations']),
            'actions': np.array(self.dataset['actions']),
            'rewards': np.array(self.dataset['rewards']),
            'next_observations': np.array(self.dataset['next_observations']),
            'terminals': np.array(self.dataset['terminals']),
            'episode_returns': np.array(self.dataset['episode_returns']),
            'episode_lengths': np.array(self.dataset['episode_lengths'])
        }
        
        dataset_file = f"{self.save_dir}/dataset/dataset_episodes_{self.episode_count}.pkl"
        with open(dataset_file, "wb") as f:
            pickle.dump(dataset_np, f)
        
        if self.verbose > 0:
            print(f"Dataset sauvegardé: {len(self.dataset['episode_returns'])} épisodes, "
                  f"{len(self.dataset['observations'])} transitions")

    def _on_training_end(self):
        """Sauvegarder le dataset final quand l'entraînement se termine."""
        self._save_dataset()
        
        # Sauvegarder le mapping vidéo-trajectoire
        mapping_file = f"{self.save_dir}/video_trajectory_mapping.pkl"
        with open(mapping_file, "wb") as f:
            pickle.dump(self.video_trajectory_map, f)
        
        print(f"\n=== Training terminé ===")
        print(f"Dataset final: {len(self.dataset['episode_returns'])} épisodes")
        print(f"Vidéos sauvegardées: {len(self.video_trajectory_map)}")
        print(f"Épisodes réussis: {sum(1 for m in self.video_trajectory_map if m['success'])}")
        print(f"Mapping vidéo-trajectoire sauvegardé: {mapping_file}")
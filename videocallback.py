import os
import pickle
import imageio
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class LoggingVideoCallback(BaseCallback):

    def __init__(self, env, save_dir="logs", video_freq=10,
                 save_success_videos=True, verbose=1, frame_skip=1):

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

        # Dataset complet (transitions)
        self.dataset = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'next_observations': [],
            'terminals': [],
            'episode_lengths': []
        }

        # Metrics par épisode
        self.metrics = {
            'episode_returns': [],
            'episode_lengths': [],
            'entropies': [],
            'value_losses': [],
            'distances_finger': [],
            'distances_target': [],
            'terminated': [],
            'truncated': []
        }

        # Map vidéo → trajectoire
        self.video_trajectory_map = []

        # Création dossiers
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(f"{save_dir}/videos", exist_ok=True)
        os.makedirs(f"{save_dir}/trajectories", exist_ok=True)
        os.makedirs(f"{save_dir}/dataset", exist_ok=True)
        os.makedirs(f"{save_dir}/metrics", exist_ok=True)


    # ------------------------------------------------------- #
    #                   ROLLOUT START                         #
    # ------------------------------------------------------- #
    def _on_rollout_start(self):
        try:
            frame = self.env.envs[0].get_frame()
            self.frames = [frame]
        except:
            self.frames = []


    # ------------------------------------------------------- #
    #                          STEP                           #
    # ------------------------------------------------------- #
    def _on_step(self) -> bool:
        env_idx = 0
        self.episode_step_count += 1

        info = self.locals["infos"][env_idx]
        terminated_flag = info["terminated"]
        truncated_flag = info["truncated"]
        done = terminated_flag or truncated_flag

        # ============================================
        #              VIDEO FRAME
        # ============================================
        if self.episode_step_count % self.frame_skip == 0:
            if hasattr(self.env.envs[env_idx], "get_frame"):
                self.frames.append(self.env.envs[env_idx].get_frame())

        # ============================================
        #               TRANSITION
        # ============================================
        obs = self.locals["new_obs"][env_idx].copy()
        action = self.locals["actions"][env_idx].copy()
        reward = self.locals["rewards"][env_idx]
        info = self.locals["infos"][env_idx]

        prev_obs = (
            self.current_episode['next_observations'][-1]
            if self.current_episode['next_observations']
            else obs
        )

        self.current_episode['observations'].append(prev_obs)
        self.current_episode['actions'].append(action)
        self.current_episode['rewards'].append(reward)
        self.current_episode['next_observations'].append(obs)
        self.current_episode['infos'].append(info)
        

        # ============================================
        #            METRICS PAR STEP
        # ============================================

        # ENTROPY
        obs_tensor, _ = self.model.policy.obs_to_tensor(np.expand_dims(obs, axis=0))
        dist = self.model.policy.get_distribution(obs_tensor)
        self.metrics['entropies'].append(dist.entropy().mean().item())


        # DISTANCE (si fournie)
        if "dist_fingers" in info:
            self.metrics['distances_finger'].append(info["dist_fingers"])
        if "dist_target" in info:
            self.metrics['distances_target'].append(info["dist_target"])

        # ============================================
        #              FIN D'ÉPISODE
        # ============================================
        if done:
            self.episode_count += 1

            # STOCKAGE terminated vs truncated
            self.metrics["terminated"].append(bool(terminated_flag))
            self.metrics["truncated"].append(bool(truncated_flag))

            # Terminals vector
            self.current_episode["terminals"] = (
                [False] * (len(self.current_episode["actions"]) - 1) + [True]
            )

            ep_return = sum(self.current_episode['rewards'])
            ep_length = len(self.current_episode['rewards'])
            is_success = info.get("dist_fingers", 1.0) < 0.085

            # Episode metrics
            self.metrics['episode_returns'].append(ep_return)
            self.metrics['episode_lengths'].append(ep_length)

            # Ajout dataset
            for key in ["observations", "actions", "rewards", "next_observations", "terminals"]:
                self.dataset[key].extend(self.current_episode[key])

            # Sauvegarde trajectoire
            traj_file = f"{self.save_dir}/trajectories/episode_{self.episode_count:04d}.pkl"
            with open(traj_file, "wb") as f:
                pickle.dump(self.current_episode, f)

            # Vidéo
            save_video = (
                self.episode_count % self.video_freq == 0
                or (self.save_success_videos and is_success)
            )
            
            if save_video and self.frames:
                if len(self.frames) < 10:
                    video_file = f"{self.save_dir}/videos/episode_{self.episode_count:04d}.mp4"
                    imageio.mimsave(video_file, self.frames, fps=1)
                    self.video_trajectory_map.append({
                        'episode': self.episode_count,
                        'video': video_file,
                        'trajectory': traj_file,
                        'return': ep_return,
                        'length': ep_length,
                        'success': is_success
                    })

            # RESET épisode
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


    # ------------------------------------------------------- #
    #                   TRAINING END                           #
    # ------------------------------------------------------- #
    def _on_training_end(self):

        # Dataset transitions
        """Sauvegarder le dataset complet."""
        dataset_np = {k: np.array(v) for k, v in self.dataset.items()}
        dataset_file = f"{self.save_dir}/dataset/dataset_final.pkl"
        with open(dataset_file, "wb") as f:
            pickle.dump(dataset_np, f)


        # Metrics
        with open(f"{self.save_dir}/metrics/metrics4.pkl", "wb") as f:
            pickle.dump(self.metrics, f)

        # Mapping video
        with open(f"{self.save_dir}/video_trajectory_mapping4.pkl", "wb") as f:
            pickle.dump(self.video_trajectory_map, f)

        print("\n=== TRAINING FINISHED ===")
        print(f"Transitions: {len(self.dataset['observations'])}")
        print(f"Episodes: {len(self.metrics['episode_returns'])}")
        print(f"Videos: {len(self.video_trajectory_map)}")

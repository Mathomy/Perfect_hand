import os
import pickle
import imageio
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

class VideoCallback(BaseCallback):
    """
    Callback for saving trajectories and videos (MP4) per episode.
    Works with Stable Baselines3 PPO and environments that implement `get_frame()`.
    """
    def __init__(self, env, save_dir="logs", log_freq=1, verbose=1, frame_skip=1):
        super().__init__(verbose)
        self.env = env
        self.save_dir = save_dir
        self.log_freq = log_freq
        self.frame_skip = frame_skip

        self.episode_count = 0
        self.episode_step_count = 0
        self.frames = []
        self.trajectory = []

        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(f"{save_dir}/videos", exist_ok=True)
        os.makedirs(f"{save_dir}/trajectories", exist_ok=True)

    def _on_step(self) -> bool:
        """
        Called at each environment step.
        """
        env_idx = 0  # assuming single environment (or DummyVecEnv)

        # Track per-episode frames
        self.episode_step_count += 1
        if self.episode_step_count % self.frame_skip == 0:
            if hasattr(self.env.envs[env_idx], "get_frame"):
                self.frames.append(self.env.envs[env_idx].get_frame())

        # Collect step info
        self.trajectory.append({
            "obs": self.locals["new_obs"][env_idx].copy(),
            "action": self.locals["actions"][env_idx].copy(),
            "reward": self.locals["rewards"][env_idx],
            "info": self.locals["infos"][env_idx],
        })

        # Check episode termination
        done = self.locals["dones"][env_idx]  # Gymnasium style
        truncated = self.locals.get("truncated", [False])[env_idx]  # handle truncation

        if done:
            self.episode_count += 1
        
            # Save trajectory
            traj_file = f"{self.save_dir}/trajectories/trajectory_{self.episode_count:04d}.pkl"
            with open(traj_file, "wb") as f:
                pickle.dump(self.trajectory, f)
            # Save video
            video_file = f"{self.save_dir}/videos/episode_{self.episode_count:04d}.mp4"
            imageio.mimsave(video_file, self.frames, fps=5)
            if self.verbose > 0:
                print(f"[Episode {self.episode_count}] Saved video + trajectory ({len(self.frames)} frames).")
        


            # Reset buffers
            self.frames = []
            self.trajectory = []
            self.episode_step_count = 0

        return True
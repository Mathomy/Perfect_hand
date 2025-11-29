import os
import pickle
import shutil
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

class DatasetFilter:
    """
    Analyze and filter saved trajectories/videos based on quality metrics.
    """
    def __init__(self, logs_dir="logs"):
        self.logs_dir = Path(logs_dir)
        self.traj_dir = self.logs_dir / "trajectories"
        self.video_dir = self.logs_dir / "videos"

        os.makedirs(self.traj_dir, exist_ok=True)
        os.makedirs(self.video_dir, exist_ok=True)
        os.makedirs(self.logs_dir / "dataset_analysis", exist_ok=True)
        os.makedirs(self.logs_dir / "dataset_analysis/trajectories_filtered", exist_ok=True)
        os.makedirs(self.logs_dir / "dataset_analysis/videos_filtered", exist_ok=True)

        self.filtered_traj_dir = self.logs_dir / "dataset_analysis/trajectories_filtered"
        self.filtered_video_dir = self.logs_dir / "dataset_analysis/videos_filtered"
        
        
    def analyze_dataset(self):
        """Analyze all trajectories and compute quality metrics."""
        traj_files = sorted(self.traj_dir.glob("trajectory_*.pkl"))
        
        results = []
        for traj_file in traj_files:
            with open(traj_file, "rb") as f:
                data = pickle.load(f)
            
            if isinstance(data, dict) and "rewards" in data:
                
                trajectory = []
                n = len(data["observations"])
                for i in range(n):
                    step = {
                        "obs": data["observations"][i],
                        "action": data["actions"][i],
                        "reward": data["rewards"][i],
                        "next_obs": data["next_observations"][i],
                        "done": data["terminals"][i],
                        "info": data["infos"][i],
                    }
                    trajectory.append(step)
                total_reward = sum(data["rewards"])
            else:
                trajectory = data
                total_reward = None

            
            # Compute metrics
            metrics = self._compute_trajectory_metrics(trajectory, total_reward)
            metrics["filename"] = traj_file.name
            results.append(metrics)
        
        return results
    
    def _compute_trajectory_metrics(self, trajectory, cached_reward=None):
        """Compute quality metrics for a single trajectory."""
        # Total reward
        if cached_reward is not None:
            total_reward = cached_reward
        else:
            total_reward = sum(step["reward"] for step in trajectory)
        
        # Final distances
        final_info = trajectory[-1]["info"]
        final_dist_fingers = final_info.get("dist_fingers", float('inf'))
        final_dist_target = final_info.get("dist_to_target", float('inf'))
        
        # Success
        success = final_info.get("success", False)
        
        # Minimum distances achieved during episode
        min_dist_fingers = min(
            step["info"].get("dist_fingers", float('inf')) 
            for step in trajectory
        )
        
        # Average reward per step
        avg_reward = total_reward / len(trajectory)
        
        return {
            "total_reward": total_reward,
            "avg_reward": avg_reward,
            "final_dist_fingers": final_dist_fingers,
            "final_dist_target": final_dist_target,
            "min_dist_fingers": min_dist_fingers,
            "success": success,
            "length": len(trajectory),
        }
    
    def filter_dataset(
        self,
        min_reward=-500,
        max_final_dist_fingers=0.05,
        max_final_dist_target=0.15,
        top_k_percent=50,
        plot=True
    ):
        """
        Filter dataset based on quality criteria.
        
        Args:
            min_reward: Minimum total reward threshold
            max_final_dist_fingers: Maximum allowed final finger distance
            max_final_dist_target: Maximum allowed final target distance
            top_k_percent: Keep only top K% by reward
            plot: Whether to plot analysis
        """
        # Analyze all trajectories
        print("Analyzing dataset...")
        results = self.analyze_dataset()
        
        if not results:
            print("No trajectories found!")
            return
        
        print(f"Found {len(results)} episodes")
        
        # Compute statistics
        rewards = [r["total_reward"] for r in results]
        print(f"\nReward statistics:")
        print(f"  Mean: {np.mean(rewards):.1f}")
        print(f"  Std: {np.std(rewards):.1f}")
        print(f"  Min: {np.min(rewards):.1f}")
        print(f"  Max: {np.max(rewards):.1f}")
        
        # Apply filters
        filtered = []
        for r in results:
            if r["total_reward"] < min_reward:
                continue
            if r["final_dist_fingers"] > max_final_dist_fingers:
                continue
            if r["final_dist_target"] > max_final_dist_target:
                continue
            filtered.append(r)
        
        # Apply percentile filter
        if filtered and top_k_percent < 100:
            threshold = np.percentile(
                [r["total_reward"] for r in filtered],
                100 - top_k_percent
            )
            filtered = [r for r in filtered if r["total_reward"] >= threshold]
        
        print(f"\nAfter filtering: {len(filtered)} episodes ({len(filtered)/len(results)*100:.1f}%)")
        
        if filtered:
            filtered_rewards = [r["total_reward"] for r in filtered]
            print(f"Filtered reward stats:")
            print(f"  Mean: {np.mean(filtered_rewards):.1f}")
            print(f"  Min: {np.min(filtered_rewards):.1f}")
            print(f"  Max: {np.max(filtered_rewards):.1f}")
        
        # Plot analysis
        if plot and results:
            self._plot_analysis(results, filtered)
        
        return filtered
    
    def _plot_analysis(self, all_results, filtered_results):
        """Plot quality analysis."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        all_rewards = [r["total_reward"] for r in all_results]
        filtered_rewards = [r["total_reward"] for r in filtered_results]
        
        # Reward distribution
        axes[0, 0].hist(all_rewards, bins=30, alpha=0.5, label="All")
        axes[0, 0].hist(filtered_rewards, bins=30, alpha=0.5, label="Filtered")
        axes[0, 0].set_xlabel("Total Reward")
        axes[0, 0].set_ylabel("Count")
        axes[0, 0].set_title("Reward Distribution")
        axes[0, 0].legend()
        
        # Final finger distance vs reward
        axes[0, 1].scatter(
            [r["total_reward"] for r in all_results],
            [r["final_dist_fingers"] for r in all_results],
            alpha=0.3, label="All"
        )
        axes[0, 1].scatter(
            [r["total_reward"] for r in filtered_results],
            [r["final_dist_fingers"] for r in filtered_results],
            alpha=0.5, label="Filtered"
        )
        axes[0, 1].set_xlabel("Total Reward")
        axes[0, 1].set_ylabel("Final Finger Distance")
        axes[0, 1].set_title("Reward vs Finger Distance")
        axes[0, 1].legend()
        
        # Final target distance vs reward
        axes[1, 0].scatter(
            [r["total_reward"] for r in all_results],
            [r["final_dist_target"] for r in all_results],
            alpha=0.3, label="All"
        )
        axes[1, 0].scatter(
            [r["total_reward"] for r in filtered_results],
            [r["final_dist_target"] for r in filtered_results],
            alpha=0.5, label="Filtered"
        )
        axes[1, 0].set_xlabel("Total Reward")
        axes[1, 0].set_ylabel("Final Target Distance")
        axes[1, 0].set_title("Reward vs Target Distance")
        axes[1, 0].legend()
        
        # Reward over episodes
        axes[1, 1].plot(all_rewards, alpha=0.5, label="All episodes")
        axes[1, 1].set_xlabel("Episode")
        axes[1, 1].set_ylabel("Total Reward")
        axes[1, 1].set_title("Reward Over Time")
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(self.logs_dir / "dataset_analysis.png", dpi=150)
        print(f"\nSaved analysis plot to {self.logs_dir / 'dataset_analysis.png'}")
        plt.show()
    
    def copy_filtered_dataset(self, filtered_results):
        """Copy filtered trajectories and videos to new directories."""
        # Create filtered directories
        self.filtered_traj_dir.mkdir(exist_ok=True)
        self.filtered_video_dir.mkdir(exist_ok=True)
        
        print(f"\nCopying {len(filtered_results)} filtered episodes...")
        
        for idx, result in enumerate(filtered_results):
            # Extract episode number from filename
            orig_name = result["filename"]
            
            # Copy trajectory
            src_traj = self.traj_dir / orig_name
            dst_traj = self.filtered_traj_dir / f"trajectory_{idx:04d}.pkl"
            shutil.copy2(src_traj, dst_traj)
            
            # Copy corresponding video
            video_name = orig_name.replace("trajectory_", "episode_").replace(".pkl", ".mp4")
            src_video = self.video_dir / video_name
            dst_video = self.filtered_video_dir / f"episode_{idx:04d}.mp4"
            
            if src_video.exists():
                shutil.copy2(src_video, dst_video)
        
        print(f"✓ Copied to {self.filtered_traj_dir} and {self.filtered_video_dir}")


if __name__ == "__main__":
    filter_tool = DatasetFilter(logs_dir="logs")
    
    # Analyze and filter
    filtered = filter_tool.filter_dataset(
        min_reward=-400,              # Require decent performance
        max_final_dist_fingers=0.04,  # Fingers must be reasonably close
        max_final_dist_target=0.15,   # Target shouldn't be too far
        top_k_percent=25,              # Keep top 25%
        plot=True
    )
    
    # Copy filtered dataset to new directory
    if filtered:
        filter_tool.copy_filtered_dataset(filtered)
    
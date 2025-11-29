from shadow_hand_reach_env import AdroitHandReachEnv
from stable_baselines3 import PPO
import mujoco
import numpy as np
from mujoco import viewer
import time

def explore_with_random_policy(log=True, steps=2000):
    env = AdroitHandReachEnv(
        fingers=["thumb", "index", "middle", "ring", "pinky"],
        target_fingers=("thumb", "index")
    )
    obs, info = env.reset()

    print("=== Random Policy Exploration ===")

    for step in range(steps):
        action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)

        if log:
            print(
                f"[{step:04d}] "
                f"dist_fingers={info['dist_fingers']:.4f} | "
                f"dist_target={info['dist_to_target']:.4f} | "
                f"reward={reward:.4f}"
            )

        if terminated or truncated:
            obs, info = env.reset()

    env.close()
    print("=== Exploration finished ===")



def visualize_random_policy():
    env = AdroitHandReachEnv( fingers=["thumb", "index", "middle", "ring", "pinky"],
    target_fingers=("thumb", "pinky"),render_mode=None
)
    obs, info = env.reset()

    with mujoco.viewer.launch_passive(env.model, env.data) as v:
        while True:
            time.sleep(0.05)
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            v.sync()

            if terminated or truncated:
                obs, info = env.reset()




if __name__ == "__main__": 
    # explore_with_random_policy()
    visualize_random_policy()
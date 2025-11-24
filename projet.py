from shadow_hand_reach_env import AdroitHandReachEnv

def main():
    env = AdroitHandReachEnv(render_mode="human")

    obs, info = env.reset()
    print("Observation shape:", obs.shape)
    print("Action space:", env.action_space)

    for t in range(300):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        if t % 20 == 0:
            print(f"Step {t} | Reward: {reward:.4f} | Distance: {info['distance']:.4f}")

        if terminated or truncated:
            print("Episode terminé, on reset.")
            obs, info = env.reset()

    env.close()

if __name__ == "__main__":
    main()

from shadow_hand_reach_env import AdroitHandReachEnv
from stable_baselines3 import PPO

def train_ppo():
    # --- Entraînement SANS rendu ---
    env_train = AdroitHandReachEnv(render_mode=None)

    model = PPO(
        policy="MlpPolicy",
        env=env_train,
        verbose=1,
        tensorboard_log="./ppo_shadowhand/"
    )

    model.learn(total_timesteps=200_000)

    model.save("ppo_shadowhand")
    env_train.close()
    print("✔ Entraînement terminé et modèle sauvegardé.")


def evaluate_model():
    # --- Évaluation AVEC rendu ---
    env_eval = AdroitHandReachEnv(render_mode="human")
    model = PPO.load("ppo_shadowhand.zip", env=env_eval)

    obs, info = env_eval.reset()

    for _ in range(2000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env_eval.step(action)

        if terminated or truncated:
            obs, info = env_eval.reset()

    env_eval.close()


if __name__ == "__main__":
    # train_ppo()      # lance l'entraînement
    evaluate_model() # décommente pour tester visuellement

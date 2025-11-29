from shadow_hand_reach_env import AdroitHandReachEnv
from stable_baselines3 import PPO
import mujoco
import numpy as np
from mujoco import viewer

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

def visualize_trained_model():
    env = AdroitHandReachEnv(render_mode=None)

    # Charge le modèle entraîné
    model = PPO.load("ppo_shadowhand.zip", env=env)

    obs, info = env.reset()

    # Ouvre le viewer passif (géré par nous)
    with mujoco.viewer.launch_passive(env.model, env.data) as v:
        try:
            while True:  # boucle infinie jusqu'à Ctrl+C
                # Action du modèle
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)

                # Debug simple si tu veux voir ce qui se passe
                # print("dist:", info.get("distance"), "reward:", reward)

                # Afficher la frame
                v.sync()

                # Si épisode fini → on reset mais on ne ferme PAS la fenêtre
                if terminated or truncated:
                    obs, info = env.reset()

        except KeyboardInterrupt:
            print("Visualisation interrompue par l'utilisateur (Ctrl+C).")

    env.close()


if __name__ == "__main__": 
    # env_train = AdroitHandReachEnv(render_mode=None) 
    # env_train.debug_actuators()
    #train_ppo() # lance l'entraînement
    visualize_trained_model() # pour la visualisation
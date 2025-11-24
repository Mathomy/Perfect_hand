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


# def evaluate_model():
#     # --- Évaluation AVEC rendu ---
#     env_eval = AdroitHandReachEnv(render_mode="human")
#     env_eval.utilis()
#     env_eval.debug_actuators()
#     model = PPO.load("ppo_shadowhand.zip", env=env_eval)

#     obs, info = env_eval.reset()
#     try : 
#         for _ in range(2000):
#             action, _ = model.predict(obs, deterministic=True)
#             print("action avant:", action)
#             obs, reward, terminated, truncated, info = env_eval.step(action)
#             print("action après", env_eval.data.qpos)

#             if terminated or truncated:
#                 obs, info = env_eval.reset()
#     finally :
#         env_eval.close()
import mujoco
import numpy as np
from mujoco import viewer


def visualize_trained_model():
    """Affiche la main qui bouge avec le modèle entraîné, via mujoco.viewer."""
    env = AdroitHandReachEnv(render_mode=None)
    model = PPO.load("ppo_shadowhand.zip", env=env)

    obs, info = env.reset()

    # # Viewer passif : on contrôle la simu nous-mêmes
    # with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
    #     for _ in range(1000):
    #         action, _ = model.predict(obs, deterministic=True)
    #         obs, reward, terminated, truncated, info = env.step(action)
    #         print("dist:", info["distance"], "reward:", reward)
    #         # Affiche un peu ce qui se passe dans le terminal si tu veux
    #         # print("action:", action, "distance:", info["distance"])

    #         viewer.sync()

    #         if terminated or truncated:
    #             obs, info = env.reset()
    with viewer.launch_passive(env.model, env.data) as v:
        for _ in range(1000):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            print(
                "dist_fingers:", info["dist_fingers"],
                "dist_to_target:", info["dist_to_target"],
                "reward:", reward
            )

            v.sync()

            if terminated or truncated:
                obs, info = env.reset()


    env.close()

if __name__ == "__main__": 
    # env_train = AdroitHandReachEnv(render_mode=None) 
    # env_train.debug_actuators()
    #train_ppo()      # lance l'entraînement
    
    #evaluate_model() # décommente pour tester visuellement
    visualize_trained_model()
    #visualize_random_actions()
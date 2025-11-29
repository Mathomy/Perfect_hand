from shadow_hand_reach_env import AdroitHandReachEnv
import numpy as np
import mujoco
import mujoco.viewer

def naive_policy(step_idx, max_steps):
    """Politique naïve qui génère les mouvement pour atteindre la position souhaité."""
    # Nb de joints par doigts
    n_index = 4
    n_thumb = 5
    n_actions = n_index + n_thumb
    action = np.zeros(n_actions)

    t = min(1.0, step_idx / max_steps ) # Normalisé entre 0 et 1 sur la durée

    # Pour l'index : on plie progressivement les 4 joints
    closing_value_index = 0.9*t
    action[0:n_index] = closing_value_index
    # Pour le pouce : on plie progressivement les 5 joints
    closing_value_thumb = 0.8*t
    action[n_index+1] = -0.5*t
    action[n_index]= -0.1*t
    action[n_index+2] = 0.5*t
    action[n_index+3] = -closing_value_thumb
    action[n_index+4] = -closing_value_thumb
    #action[n_index:n_index+n_thumb] = closing_value_thumb
    action[n_index+1] = closing_value_thumb
    #action[n_index] = -0.5*t
    #action[n_index+n_thumb-1] = -0.5*t  # Ouvrir un peu l'articulation finale du pouce
    action = np.clip(action, -1, 1) # S'assure que l'action est dans les bornes

    return action


def run_naive():
    env = AdroitHandReachEnv(render_mode=None)

    obs, info = env.reset()
    max_steps = 1000
    
    with mujoco.viewer.launch_passive(env.model, env.data) as v:
        try:
            # last_action = None
            step_idx = 0
            while True:
                action = naive_policy(step_idx, max_steps)
                obs, reward, terminated, truncated, info = env.step(action)
                #fb = naive_feedback(obs,info,last_action)
                #last_action = action
                # print pour debug
                if step_idx % 10 == 0:
                    print(f"Step {step_idx}: dist_fingers={info['dist_fingers']:.4f}, reward={reward:.4f}, info={info}")
                v.sync()

                step_idx += 1
                if step_idx >= max_steps :
                    pass
                if terminated or truncated:
                    obs, info = env.reset()
                    step_idx = 0
        except KeyboardInterrupt:
            print("Simulation interrompue par l'utilisateur (Ctrl+C).")
    env.close()


# def naive_feedback(obs,info,last_action):
#     """Politique naïve avec feedback pour ajuster les mouvements en fonction de la distance aux doigts."""
#     if last_action is None:
#         last_action = np.zeros(9)
#     dist = info.get('distance', 0.0)
#     action = last_action.copy()

#     if dist is None:
#         action += 0.01  # fermer doucement
#     else:
#         if dist > 0.04:
#             action += 0.02  # fermer plus rapidement
#         elif dist > 0.02:
#             action += 0.005  # fermer doucement
#         else:
#             action[:]=0.0
#     action = np.clip(action, -1, 1) # S'assure que l'action est dans les bornes
#     return action


if __name__ == "__main__": 
    # env_train = AdroitHandReachEnv(render_mode=None) 
    # env_train.debug_actuators()
    run_naive()
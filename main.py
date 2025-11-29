from stable_baselines3 import PPO
from adroit_env_datafocused import AdroitTrajEnv
from datasets import annoter_et_mettre_score, generer_clips, generer_paires_aleatoires, charger_episodes
from shadow_hand_reach_env import AdroitHandReachEnv
import time
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from videocallback import LoggingVideoCallback
from PPO_policyv2 import train_ppo, visualize_trained_model, plot_and_save_metrics
from random_policy import visualize_random_policy 
from explore import train_exploration_sac
from naive_policy import run_naive
from PbRL import train_with_pbrl,visualize_trained_model,evaluate_model

from filtered_data import DatasetFilter


if __name__ == "__main__":

    """ Example usages """
    # Exemple : entraînement court pour debug
    # model = train_exploration_sac(total_timesteps=5000, k=5, buffer_size=20, seed=1)
    # # Visualiser quelques seconds
    # visualize_policy(model_path="sac_explore_adroit.zip", n_steps=1000)
     # model PPO
    #train_ppo() 
    # visualize_trained_model()
    # with open("C:/Users/tlamy/Sorbonne/Social robotic/Perfect_hand/logs/metrics/metrics4.pkl", "rb") as f:
    #     metrics = pickle.load(f)
    # with open("C:/Users/tlamy/Sorbonne/Social robotic/Perfect_hand/logs/dataset/dataset_final.pkl", "rb") as f:
    #     dataset = pickle.load(f)
    # plot_and_save_metrics(dataset,metrics)
    #model random
    # visualize_random_policy () 
    #model naive
    #run_naive()


   
    #----------------------------------#
    #     Full pipeline execution      #
    #----------------------------------#       

    """ PPO trainaing to generate dataset"""
    train_ppo() 
 
    """ Filtering trajectories only if necessary """
    """ 
    filter_tool = DatasetFilter(logs_dir="logs")
    
    #Analyze and filter
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
    """

    """ Dataset creation for PbRL """
    # Define Folders

    #For filtered data
    #dossier_traj = "logs/dataset_analysis/trajectories_filtered"
    #dossier_videos = "logs/dataset_analysis/videos_filtered"

    #For unfiltered data
    dossier_traj = "logs/trajectories"
    dossier_videos = "logs/videos"
  

    # 1. Charger les épisodes
    episodes = charger_episodes(dossier_traj, dossier_videos)

    # 2. Générer les clips avec ID unique
    clips_dict = generer_clips(episodes, fps=30, duree_clip=1.5)

    # 3. Générer autant de paires que nécessaire (ici par exemple 200)
    paires = generer_paires_aleatoires(clips_dict, max_paires=200)

    # 4. Annoter les paires et mettre à jour les scores
    clips_dict, preference_data = annoter_et_mettre_score(clips_dict, paires, fps=30)

    # 5. Sauvegarder le dataset final
    with open("logs/dataset/preference_dataset.pkl", "wb") as f:
        pickle.dump(clips_dict, f)
    

    with open("logs/dataset/sigma_data.pkl", "wb") as f:
        pickle.dump(preference_data, f)

    print("Dataset de préférence sauvegardé !")

    """Training with PbRL """
    train_with_pbrl() 
    evaluate_model()
    visualize_trained_model()
    


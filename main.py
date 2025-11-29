from stable_baselines3 import PPO
from adroit_env_datafocused import AdroitTrajEnv
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


if __name__ == "__main__":
    # Exemple : entraînement court pour debug
    # model = train_exploration_sac(total_timesteps=5000, k=5, buffer_size=20, seed=1)
    # # Visualiser quelques seconds
    # visualize_policy(model_path="sac_explore_adroit.zip", n_steps=1000)
     # model PPO
    # train_ppo() 
    # visualize_trained_model()
    # with open("C:/Users/tlamy/Sorbonne/Social robotic/Perfect_hand/logs/metrics/metrics4.pkl", "rb") as f:
    #     metrics = pickle.load(f)
    # with open("C:/Users/tlamy/Sorbonne/Social robotic/Perfect_hand/logs/dataset/dataset_final.pkl", "rb") as f:
    #     dataset = pickle.load(f)
    # plot_and_save_metrics(dataset,metrics)
    #model random
    # visualize_random_policy () 
    #model naive
    run_naive()
    #human reinforcement_learning
    # train_with_pbrl()
    # visualize_trained_model()
    # evaluate_model()


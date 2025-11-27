import os
import pickle
import random
import math
import imageio
import cv2
import numpy as np

# dans le .pkl il y a : états, actions, rewards

def liste_traj(chemin):
    """
    Charge toutes les trajectoires .pkl d'un dossier et les retourne sous forme de liste.
    
    Args:
        chemin (str): Chemin vers le dossier contenant les fichiers .pkl.
        
    Returns:
        list: Liste de dictionnaires, chaque dictionnaire correspondant à une trajectoire.
    """
    # Lister et trier tous les fichiers .pkl
    pkl_files = [f for f in os.listdir(chemin) if f.endswith(".pkl")]
    pkl_files.sort()  # Tri alphabétique
    
    trajectoires = []
    for file_name in pkl_files:
        file_path = os.path.join(chemin, file_name)
        with open(file_path, "rb") as f:
            traj = pickle.load(f)
            trajectoires.append(traj)
    
    return trajectoires


def charger_episodes(dossier_traj, dossier_videos):
    """
    Charge toutes les trajectoires et associe les vidéos correspondantes.
    
    Args:
        dossier_traj (str): Chemin vers le dossier contenant les fichiers .pkl des trajectoires
        dossier_videos (str): Chemin vers le dossier contenant les fichiers .mp4 des vidéos

    Returns:
        list: Liste de dictionnaires, un par épisode, avec les informations suivantes :
            - 'numero_episode' : numéro de l'épisode (str)
            - 'trajectoire'    : dictionnaire de la trajectoire (observations, actions, rewards, etc.)
            - 'chemin_video'   : chemin vers la vidéo correspondante (str ou None si pas de vidéo)
            - 'retour'         : somme des rewards de l'épisode
            - 'succes'         : booléen indiquant si l'épisode est considéré comme réussi
    """
    
    fichiers_traj = sorted([f for f in os.listdir(dossier_traj) if f.endswith(".pkl")])
    
    fichiers_videos = []
    if dossier_videos is not None:
        fichiers_videos = sorted([f for f in os.listdir(dossier_videos) if f.endswith(".mp4")])
    
    episodes = []
    
    for fichier_traj in fichiers_traj:
        chemin_traj = os.path.join(dossier_traj, fichier_traj)
        with open(chemin_traj, "rb") as f:
            trajectoire = pickle.load(f)

        # Extraire le numéro de l'épisode
        numero_episode = fichier_traj.split("_")[1].split(".")[0]

        # Chercher la vidéo correspondante si le dossier est fourni
        chemin_video = None
        if fichiers_videos:
            for fichier_video in fichiers_videos:
                if numero_episode in fichier_video:
                    chemin_video = os.path.join(dossier_videos, fichier_video)
                    break

        episodes.append({
            "numero_episode": numero_episode,
            "trajectoire": trajectoire,
            "chemin_video": chemin_video,  # None si pas de vidéo
            "retour": sum(trajectoire['rewards']),
            "succes": sum(trajectoire['rewards']) > -100
        })
        
    return episodes

def generer_clips(episodes, fps=30, duree_clip=1.5):
    """
    Génère tous les clips à partir des épisodes avec vidéo, avec ID unique.

    Returns:
        clips_dict (dict): {clip_id: {'video_frames':..., 'traj_segment':..., 'score_preference':0}}
    """
    clips_dict = {}
    clip_counter = 0

    episodes_vid = [ep for ep in episodes if ep['chemin_video'] is not None]

    for ep in episodes_vid:
        video_path = ep['chemin_video']
        traj = ep['trajectoire']
        reader = imageio.get_reader(video_path)
        nb_frames = reader.count_frames()
        frames_per_clip = int(fps * duree_clip)

        for start in range(0, nb_frames, frames_per_clip):
            end = min(start + frames_per_clip, nb_frames)
            clip_frames = [reader.get_data(i) for i in range(start, end)]
            
            traj_segment = {
                'observations': traj['observations'][start:end],
                'actions': traj['actions'][start:end],
                'rewards': traj['rewards'][start:end],
                'score_preference': 0
            }

            clip_id = f"clip_{clip_counter}"
            clips_dict[clip_id] = {
                'video_frames': clip_frames,
                'traj_segment': traj_segment
            }
            clip_counter += 1

        reader.close()
    
    return clips_dict


def generer_paires_aleatoires(clips_dict, max_paires=100):
    """
    Génère des paires aléatoires à partir du dictionnaire de clips.
    """
    clip_ids = list(clips_dict.keys())
    paires = []

    for _ in range(min(max_paires, len(clip_ids)*(len(clip_ids)-1)//2)):
        clip1, clip2 = random.sample(clip_ids, 2)
        paires.append((clip1, clip2))  # On stocke juste les IDs
    return paires

def annoter_et_mettre_score(clips_dict, paires, fps=30):
    delay = int(1000 / fps)

    for i, (clip1_id, clip2_id) in enumerate(paires):
        clip1_frames = clips_dict[clip1_id]['video_frames']
        clip2_frames = clips_dict[clip2_id]['video_frames']
        nb_frames = max(len(clip1_frames), len(clip2_frames))

        for j in range(nb_frames):
            frame1 = clip1_frames[j] if j < len(clip1_frames) else clip1_frames[-1]
            frame2 = clip2_frames[j] if j < len(clip2_frames) else clip2_frames[-1]

            frame1 = cv2.resize(frame1, (400, 400))
            frame2 = cv2.resize(frame2, (400, 400))

            cv2.imshow("Clip1", frame1)
            cv2.imshow("Clip2", frame2)

            if cv2.waitKey(delay) & 0xFF == ord('q'):
                break

        pref = None
        while pref not in ['1', '2', 's']:
            pref = input(f"Paire {i+1}/{len(paires)} - tapez 1 ou 2 pour le clip préféré (s pour sauter): ")

        if pref == '1':
            clips_dict[clip1_id]['traj_segment']['score_preference'] += 1
        elif pref == '2':
            clips_dict[clip2_id]['traj_segment']['score_preference'] += 1
        # 's' => pas de score

    cv2.destroyAllWindows()
    return clips_dict



 #### TEST création du dataset

# Définir les dossiers
dossier_traj = "logs/trajectories"
dossier_videos = "logs/videos"

# 1. Charger les épisodes
episodes = charger_episodes(dossier_traj, dossier_videos)

# 2. Générer les clips avec ID unique
clips_dict = generer_clips(episodes, fps=30, duree_clip=1.5)

# 3. Générer autant de paires que nécessaire (ici par exemple 200)
paires = generer_paires_aleatoires(clips_dict, max_paires=10)

# 4. Annoter les paires et mettre à jour les scores
clips_dict = annoter_et_mettre_score(clips_dict, paires, fps=30)

# 5. Sauvegarder le dataset final
with open("logs/dataset/preference_dataset.pkl", "wb") as f:
    pickle.dump(clips_dict, f)

print("Dataset de préférence sauvegardé !")



### Pour voir le score de chaque clip
""" 
with open("logs/dataset/preference_dataset.pkl", "rb") as f:
    data = pickle.load(f)

# Afficher le score de chaque clip
for i in range(len(data)):
    clip_id = f"clip_{i}"
    if clip_id in data:
        score = data[clip_id]["traj_segment"]["score_preference"]
        print(clip_id, "→ score =", score)
    else:
        print(clip_id, "absent du dataset") """

# import pickle
# import numpy as np

# with open("logs/dataset/preference_dataset.pkl", "rb") as f:
#     data = pickle.load(f)

# for clip_id in data.keys():
#     print("\n----", clip_id, "----")
#     seg = data[clip_id]["traj_segment"]

#     print("Nombre d'observations :", len(seg["observations"]))
#     print("Observation[0] (shape) :", np.array(seg["observations"][0]).shape)

#     print("Nombre d'actions :", len(seg["actions"]))
#     print("Action[0] :", seg["actions"][0])

#     print("Rewards totaux du clip :", sum(seg["rewards"]))
#     print("Score de préférence :", seg["score_preference"])

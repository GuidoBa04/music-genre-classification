import os
import numpy as np
import librosa
import argparse
from tqdm import tqdm
from utils import load_tracks, get_audio_path


#préprocessing.py
# transforme l'onde sonore mp3 en un Mel-spectrogramme https://huggingface.co/learn/audio-course/fr/chapter1/preprocessing

# Paramètres audio (Standard FMA)
SR = 22050
DURATION = 29.9 # FMA coupe parfois un peu avant 30s
N_MELS = 128
HOP_LENGTH = 512
SAMPLES = int(SR * DURATION)

def compute_mel_spectrogram(audio_path):
    try:
        # 1. Chargement audio (mono par défaut)
        y, sr = librosa.load(audio_path, sr=SR, duration=DURATION)
        
        # Padding si le fichier est trop court
        if len(y) < SAMPLES:
            padding = SAMPLES - len(y)
            y = np.pad(y, (0, padding), mode='constant')
        else:
            y = y[:SAMPLES]
            
        # 2. Calcul du Mel-Spectrogramme
        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=N_MELS, hop_length=HOP_LENGTH
        )
        
        # 3. Conversion en décibels (Log-Mel)
        mel_spec_db = librosa.feature.melspectrogram(
            S=mel_spec, # Optimization: pass S directly if needed or compute db
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # 4. Normalisation "Per Track" (Mean=0, Std=1)
        mean = mel_spec_db.mean()
        std = mel_spec_db.std()
        if std > 0:
            mel_spec_norm = (mel_spec_db - mean) / std
        else:
            mel_spec_norm = mel_spec_db
            
        return mel_spec_norm.astype(np.float32)
        
    except Exception as e:
        print(f"Erreur sur {audio_path}: {e}")
        return None

def process_dataset(data_root):
    raw_dir = os.path.join(data_root, 'raw', 'fma_small')
    meta_path = os.path.join(data_root, 'metadata', 'tracks.csv')
    processed_dir = os.path.join(data_root, 'processed')
    
    os.makedirs(processed_dir, exist_ok=True)
    
    print("Chargement des métadonnées...")
    tracks = load_tracks(meta_path)
    
    print(f"Traitement de {len(tracks)} pistes...")
    
    for track_id, _ in tqdm(tracks.iterrows(), total=len(tracks)):
        audio_path = get_audio_path(raw_dir, track_id)
        save_path = os.path.join(processed_dir, f"{track_id}.npy")
        
        # Si le fichier existe déjà, on passe (utile si le script plante)
        if os.path.exists(save_path):
            continue
            
        if os.path.exists(audio_path):
            mel = compute_mel_spectrogram(audio_path)
            if mel is not None:
                np.save(save_path, mel)

if __name__ == "__main__":
    # Ajuste ce chemin selon ton execution
    # Si tu lances depuis la racine du projet :
    DATA_ROOT = "data" 
    process_dataset(DATA_ROOT)
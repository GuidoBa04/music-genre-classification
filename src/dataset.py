import os
import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from utils import load_tracks

class FMA2DDataset(Dataset):
    def __init__(self, data_root, split='training'):
        self.data_root = data_root
        self.processed_dir = os.path.join(data_root, 'processed')
        
        # Chargement et filtrage des métadonnées
        tracks = load_tracks(os.path.join(data_root, 'metadata', 'tracks.csv'))
        self.tracks = tracks[tracks['split'] == split]
        
        self.existing_tracks = []
        self.labels = []
        
        # Vérification physique des fichiers
        print(f"Vérification des fichiers pour le split '{split}'...")
        for track_id, row in self.tracks.iterrows():
            file_path = os.path.join(self.processed_dir, f"{track_id}.npy")
            if os.path.exists(file_path):
                self.existing_tracks.append(file_path)
                self.labels.append(row['genre'])
        
        # Encodage des genres (Rock -> 0, Hip-Hop -> 1, etc.)
        # Note: Pour un vrai projet, on fitterait le LabelEncoder sur tout le dataset, 
        # mais ici les genres sont équilibrés dans chaque split.
        self.le = LabelEncoder()
        self.encoded_labels = self.le.fit_transform(self.labels)
        self.classes = self.le.classes_
        
        print(f"Dataset '{split}' prêt : {len(self.existing_tracks)} pistes.")

    def __len__(self):
        return len(self.existing_tracks)

    def __getitem__(self, idx):
        # Charge le spectrogramme (128, Time)
        mel_spec = np.load(self.existing_tracks[idx])
        
        # Ajoute la dimension Channel -> (1, 128, Time)
        mel_spec = mel_spec[np.newaxis, ...]
        
        # Conversion en Tensor PyTorch
        return torch.from_numpy(mel_spec).float(), torch.tensor(self.encoded_labels[idx]).long()
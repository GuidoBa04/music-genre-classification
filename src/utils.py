import pandas as pd
import os
import ast

def load_tracks(metadata_path):
    """
    Charge le fichier tracks.csv en gérant le multi-index header.
    """
    tracks = pd.read_csv(metadata_path, index_col=0, header=[0, 1])
    
    # Filtrer pour ne garder que le subset 'small'
    COL_SUBSET = ('set', 'subset')
    tracks = tracks[tracks[COL_SUBSET] == 'small']
    
    # Garder uniquement la colonne du genre et le split (training/validation/test)
    tracks = tracks[[('track', 'genre_top'), ('set', 'split')]]
    
    # Aplatir les colonnes pour plus de facilité
    tracks.columns = ['genre', 'split']
    
    return tracks

def get_audio_path(audio_dir, track_id):
    """
    Retourne le chemin complet d'un fichier audio donné son ID.
    Ex: track_id 2 -> .../000/000002.mp3
    """
    tid_str = '{:06d}'.format(track_id)
    return os.path.join(audio_dir, tid_str[:3], tid_str + '.mp3')
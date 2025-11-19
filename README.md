# Deep Learning for Music Genre Classification

**Auteur :** Baptiste Guido  
**Date :** Octobre 2025  
**Contexte :** Projet académique de Deep Learning

---

## 0. Installation et Reproduction

### Environnement
```bash
pip install -r requirements.txt
```

### Exécution du Prétraitement

Lancer le script pour générer les spectrogrammes dans `data/processed/` :
```bash
python src/preprocessing.py
```

**Note :** Le script gère automatiquement les erreurs de décodage MP3 inhérentes au dataset FMA original.

## 1. Vue d'ensemble

Ce projet vise à implémenter, entraîner et évaluer un modèle de Deep Learning (CNN) pour la classification automatique de genres musicaux. Le projet utilise le dataset **FMA-small** (Free Music Archive), composé de 8 000 pistes audio réparties équitablement sur 8 genres.

L'approche repose sur la conversion de signaux audio bruts en représentations temps-fréquence (Mel-spectrogrammes), qui servent ensuite d'entrée à un Réseau de Neurones Convolutif (CNN) 2D.

---

## 2. Structure du Projet
```text
music-genre-classification/
│
├── data/
│   ├── raw/fma_small/       # Fichiers audio bruts (.mp3)
│   ├── processed/           # Spectrogrammes traités (.npy)
│   └── metadata/            # Fichiers CSV (tracks.csv, etc.)
│
├── src/
│   ├── utils.py             # Gestion des métadonnées et I/O
│   ├── preprocessing.py     # Pipeline de transformation audio
│   ├── dataset.py           # Classe Dataset PyTorch
│   ├── model.py             # Architecture du CNN
│   └── train.py             # Boucle d'entraînement
│
├── requirements.txt         # Dépendances Python
└── README.md                # Documentation du projet
```

---

## 3. Pipeline de Traitement des Données

La préparation des données est une étape critique divisée en deux modules principaux : la gestion des métadonnées et le traitement du signal.

### 3.1 Gestion des Métadonnées (`src/utils.py`)

Le fichier de métadonnées `tracks.csv` du dataset FMA présente une structure complexe avec des en-têtes multi-indexés. Le module `utils.py` assure :

- **Le chargement structuré :** Parsing du CSV avec gestion des multi-headers (`set`, `subset`) et (`track`, `genre_top`).
- **Le filtrage :** Extraction stricte du sous-ensemble `small` pour garantir l'utilisation des 8 000 pistes équilibrées.
- **L'alignement :** Association de l'identifiant unique de chaque piste (`track_id`) avec son chemin de fichier et son étiquette de genre.

### 3.2 Prétraitement Audio (`src/preprocessing.py`)

Ce module convertit les fichiers audio compressés (`.mp3`) en tenseurs prêts pour l'apprentissage. Le pipeline applique les transformations suivantes :

#### Chargement et Rééchantillonnage

- **Fréquence d'échantillonnage (SR) :** Fixée à 22 050 Hz.
  - *Justification technique :* Suffisant pour couvrir le spectre utile à la classification de genre (bande passante jusqu'à ~11 kHz) tout en optimisant la mémoire et le temps de calcul par rapport au standard CD (44.1 kHz).

- **Durée :** Standardisée à 29.9 secondes.
  - *Justification technique :* Cette durée légèrement inférieure à 30s permet de gérer la variabilité de longueur due à l'encodage MP3, évitant un zero-padding systématique sur des fichiers de 29.98s. Les fichiers plus courts sont complétés par des zéros.

#### Extraction de Caractéristiques (Feature Extraction)

- **Calcul du Mel-Spectrogramme :** Transformation de Fourier à court terme (STFT) projetée sur l'échelle de Mel.
  - **Paramètres :**
    - `n_mels` : 128 bandes de fréquences (conforme aux spécifications du projet).
    - `n_fft` : 2048 (fenêtre d'analyse d'environ 93ms).
    - `hop_length` : 512 (décalage temporel).

- **Conversion Logarithmique :** Passage en échelle décibel (Log-Mel) pour correspondre à la perception logarithmique de l'intensité sonore par l'oreille humaine.

#### Normalisation (Per-Track)

Chaque spectrogramme est normalisé individuellement selon la méthode Z-score (Standard Score) :

$$X_{norm} = \frac{X - \mu}{\sigma}$$

Cela garantit que les données d'entrée du réseau ont une moyenne de 0 et un écart-type de 1, facilitant la convergence de l'optimiseur.

#### Stockage

Les matrices résultantes (shape: 128 × 1288) sont sauvegardées au format binaire NumPy (`.npy`). Cela réduit les opérations I/O et la charge CPU durant l'entraînement.

---
## 4. Architecture du Modèle (Baseline)

Le modèle implémenté dans `src/model.py` est un CNN 2D standard à 4 blocs de convolution, conçu pour servir de point de comparaison.

### Architecture détaillée

#### 1. Blocs Convolutifs (×4)

- **`Conv2d`** (Kernel 3×3, Padding 1) : Extraction de features locales.
- **`ReLU`** : Activation non-linéaire.
- **`MaxPool2d`** (2×2) : Réduction de la dimensionnalité spatiale.
- **`Dropout`** (0.1) : Légère régularisation.
- **Filtres successifs :** 16 → 32 → 64 → 128.

#### 2. Classification

- **`Flatten`** : Aplatissement des features maps.
- **`Linear`** : Couche dense connectée aux 8 classes de sortie.

---

## 5. Entraînement (`src/train.py`)

- **Optimiseur :** Adam (Learning Rate = 0.001).
- **Fonction de Coût :** Cross Entropy Loss.
- **Stratégie :** 20 Époques avec sauvegarde automatique du modèle ayant la meilleure précision de validation (Best Model Checkpoint).
- **Hardware :** Support de l'accélération GPU (CUDA).

---

## 6. Résultats et Analyse (Baseline)

Le modèle Baseline a été entraîné et évalué sur le jeu de test.

### 6.1 Performances

- **Training Accuracy :** ~96%
- **Validation/Test Accuracy :** ~39%

### 6.2 Analyse de l'Overfitting

Un phénomène de surapprentissage (overfitting) massif est observé. Dès la 5ème époque, la perte d'entraînement continue de diminuer drastiquement tandis que la précision de validation stagne autour de 40-45%. Le modèle mémorise le bruit du jeu d'entraînement au lieu de généraliser.

### 6.3 Analyse de la Matrice de Confusion

L'analyse des erreurs sur le jeu de test révèle des disparités fortes entre les genres :

- **Genres bien classés :** Hip-Hop (F1-score: 0.65) et Rock (F1-score: 0.53). Ces genres possèdent des signatures spectrales distinctes (basses fréquences, distorsion).

- **Confusions majeures :**
  - La classe **Instrumental** agit comme un "puits" attracteur : une grande partie des morceaux Folk, Experimental et Pop sont incorrectement prédits comme Instrumental.
  - La distinction entre **Pop** et **International** est particulièrement faible.

### 6.4 Pistes d'amélioration (Objectifs futurs)

Pour dépasser cette baseline, l'objectif principal sera la **régularisation du modèle** pour combattre l'overfitting. La piste privilégiée est l'ajout de **Batch Normalization** pour stabiliser les activations internes du réseau.

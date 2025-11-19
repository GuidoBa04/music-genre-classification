import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader
from dataset import FMA2DDataset
from model import BaselineCNN

# Configuration
DATA_ROOT = "data"
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate():
    print("Chargement des données de TEST...")
    test_dataset = FMA2DDataset(DATA_ROOT, split='test')
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Recharger le modèle
    # Note : On doit réinstancier la classe avec les mêmes paramètres
    # On prend un sample pour avoir la bonne input_shape
    sample_data, _ = test_dataset[0]
    model = BaselineCNN(num_classes=8, input_shape=sample_data.shape).to(DEVICE)
    
    # Charger les poids du meilleur modèle sauvegardé
    try:
        model.load_state_dict(torch.load("best_model.pth"))
        print("Modèle 'best_model.pth' chargé avec succès.")
    except FileNotFoundError:
        print("Erreur : Le fichier 'best_model.pth' est introuvable. Lancez train.py d'abord.")
        return

    model.eval()
    all_preds = []
    all_labels = []
    
    print("Lancement de l'évaluation...")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    # --- Analyse des résultats ---
    
    # 1. Rapport de classification (Précision par genre)
    print("\n--- Rapport de Classification ---")
    print(classification_report(all_labels, all_preds, target_names=test_dataset.classes))
    
    # 2. Matrice de Confusion
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=test_dataset.classes, 
                yticklabels=test_dataset.classes)
    plt.xlabel('Prédiction')
    plt.ylabel('Vérité')
    plt.title('Matrice de Confusion - Baseline')
    plt.show()

if __name__ == "__main__":
    evaluate()
import torch
import torch.nn as nn

class BaselineCNN(nn.Module):
    def __init__(self, num_classes=8, input_shape=(1, 128, 1288)):
        super(BaselineCNN, self).__init__()
        
        # 4 Blocs de Convolution comme demandé
        self.layer1 = self._make_block(1, 16)
        self.layer2 = self._make_block(16, 32)
        self.layer3 = self._make_block(32, 64)
        self.layer4 = self._make_block(64, 128)
        
        self.flatten = nn.Flatten()
        
        # Calcul automatique de la taille pour la couche dense
        flat_size = self._get_flatten_size(input_shape)
        self.fc = nn.Linear(flat_size, num_classes)

    def _make_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.1)
        )

    def _get_flatten_size(self, shape):
        with torch.no_grad():
            dummy = torch.zeros(1, *shape)
            x = self.layer1(dummy)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            return x.view(1, -1).size(1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.flatten(x)
        return self.fc(x)
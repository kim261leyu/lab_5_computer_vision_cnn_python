import torch.nn as nn
import torch

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1024, 10)
        )

    def forward(self, x):
        return self.net(x)


class MultiDigitCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(5120, 50)
        )

    def forward(self, x):
        return self.net(x).view(-1, 5, 10)
        

class CRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            # TODO: ReLU, MaxPool(2), Conv(16 -> 32), ReLU, MaxPool(2)
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.lstm = nn.LSTM(256, 64, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(128, 11)

    def forward(self, x):
        x = self.cnn(x)              # [batch, 32, 8, 40]
        x = x.permute(0, 3, 1, 2)
        x = x.flatten(2)
        # TODO: permute so width comes first, then flatten the last two dims -> [batch, 40, 256]
        x, _ = self.lstm(x)          # [batch, 40, 128]
        return self.fc(x)            # [batch, 40, 11]
      
if __name__ == "__main__":
    model = CRNN()
    print(sum(p.numel() for p in model.parameters()))   # 171083
    x = torch.rand(64, 1, 32, 160)
    print(model(x).shape)                               # torch.Size([64, 40, 11])
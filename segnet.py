
import torch
import torch.nn as nn
import torch.nn.functional as F

num_classes = 6

class SegNet(nn.Module):
    def __init__(self):
        super(SegNet, self).__init__()
        # Encoder layers
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1), # Assuming input has 1 channel
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(2, 2, return_indices=True)

        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(2, 2, return_indices=True)

        # Decoder layers
        self.dec1 = nn.Sequential(
            nn.MaxUnpool2d(2, 2),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.dec2 = nn.Sequential(
            nn.MaxUnpool2d(2, 2),
            nn.Conv2d(64, num_classes, kernel_size=3, padding=1) # output layers for num_classes
        )

    def forward(self, x):
        x, ind1 = self.pool1(self.enc1(x))
        x, ind2 = self.pool2(self.enc2(x))
        x = self.dec1((x, ind2))
        x = self.dec2((x, ind1))
        return x

def test():
    x = torch.randn((3, 1, 256, 256))  # Changing test input to match your dataset resolution
    model = SegNet()
    preds = model(x)
    print(preds.shape)
    print(x.shape)
    assert preds.shape == (3, num_classes, 256, 256), "Output shape mismatch."
    print("Success!!!")

if __name__ == "__main__":
    test()
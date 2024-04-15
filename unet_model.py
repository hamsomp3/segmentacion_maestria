# https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/ original paper link

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
#pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

num_classes = 6
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), # Batch Normalization
            nn.ReLU(inplace=True), # inplace=True -> input directly modified
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), # Batch Normalization
            nn.ReLU(inplace=True), # inplace=True -> input directly modified
        )   

    def forward(self, x):
        # return self.double_conv(x)
        return self.conv(x)

class UNET(nn.Module):
    def __init__(self, in_channels=3, out_channels=num_classes, features=None):
        if features is None:
            features = [64, 128, 256, 512]
        super(UNET,self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of UNET   
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature*2, feature))

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)

        # Final Convolution
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]
            if x.size() != skip_connection.size():
                x = TF.resize(x, size=skip_connection.shape[2:])
            x = torch.cat((x, skip_connection), dim=1)
            x = self.ups[idx + 1](x)
        
        x = self.final_conv(x)
        return torch.softmax(x, dim=1)
    
def test():
        x = torch.randn((3, 1, 160, 160))
        model = UNET(in_channels=1, out_channels=num_classes)
        preds = model(x)
        print(preds.shape)
        print(x.shape)
        assert preds.shape == (3, 5, 160, 160)
        print("Success!!!")
    
if __name__ == "__main__":
        test()
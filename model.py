import torch
import torch.nn as torch_nn
import torchvision.models as models
import torch.nn.functional as F

class DocumentCornerNet(torch_nn.Module):
    def __init__(self, pretrained=True):
        super(DocumentCornerNet, self).__init__()
        
        # Use MobileNetV3-Small for fast mobile inference
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        mobilenet = models.mobilenet_v3_small(weights=weights)
        
        # Extract features (the backbone)
        # MobileNetV3-Small features output 576 channels at 1/32 resolution
        self.backbone = mobilenet.features
        
        # Lightweight Upsampling Head to generate heatmaps
        # We need to upsample from 1/32 to maybe 1/4 or 1/8 resolution
        # Let's upsample to 1/4 resolution for the heatmaps.
        
        self.up1 = torch_nn.Sequential(
            torch_nn.ConvTranspose2d(576, 256, kernel_size=4, stride=2, padding=1, bias=False),
            torch_nn.BatchNorm2d(256),
            torch_nn.ReLU(inplace=True)
        ) # 1/16 resolution
        
        self.up2 = torch_nn.Sequential(
            torch_nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            torch_nn.BatchNorm2d(128),
            torch_nn.ReLU(inplace=True)
        ) # 1/8 resolution
        
        self.up3 = torch_nn.Sequential(
            torch_nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            torch_nn.BatchNorm2d(64),
            torch_nn.ReLU(inplace=True)
        ) # 1/4 resolution
        
        # Final prediction head: 4 channels (one for each corner)
        # 0: Top-Left, 1: Top-Right, 2: Bottom-Right, 3: Bottom-Left
        self.head = torch_nn.Conv2d(64, 4, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # x: [B, 3, H, W]
        features = self.backbone(x)
        
        x = self.up1(features)
        x = self.up2(x)
        x = self.up3(x)
        
        heatmaps = self.head(x)
        # heatmaps: [B, 4, H/4, W/4]
        
        # We use a sigmoid to constrain the heatmap values between 0 and 1
        return torch.sigmoid(heatmaps)

if __name__ == '__main__':
    # Quick test to verify model shape
    model = DocumentCornerNet(pretrained=False)
    dummy_input = torch.randn(1, 3, 256, 256)
    out = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {out.shape}") # Expected: [1, 4, 64, 64]

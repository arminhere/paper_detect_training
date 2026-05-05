import os
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class DocumentCornerDataset(Dataset):
    def __init__(self, json_path, img_dir, img_size=256, heatmap_size=64, sigma=3.0, transform=None):
        """
        Args:
            json_path: Path to the labels.json file
            img_dir: Directory with all the images
            img_size: Size of the input image (H, W)
            heatmap_size: Size of the target heatmap (H, W)
            sigma: Standard deviation for the Gaussian blob
            transform: Optional albumentations transform to be applied on a sample.
        """
        self.img_dir = img_dir
        with open(json_path, 'r') as f:
            self.labels = json.load(f)
            
        self.img_size = img_size
        self.heatmap_size = heatmap_size
        self.sigma = sigma
        self.transform = transform
        
        # Default transforms if none provided (normalize to ImageNet stats)
        if self.transform is None:
            self.transform = A.Compose([
                A.Resize(height=img_size, width=img_size),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ], keypoint_params=A.KeypointParams(format='xy'))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = self.labels[idx]
        img_path = os.path.join(self.img_dir, item['image'])
        
        # Read image
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Original coordinates
        keypoints = item['corners']
        
        # Apply transforms (resizes image and keypoints, and normalizes)
        transformed = self.transform(image=image, keypoints=keypoints)
        image = transformed['image']
        transformed_keypoints = transformed['keypoints']
        
        # Generate heatmaps
        # We need to scale the transformed keypoints (which are on img_size scale) 
        # down to the heatmap_size scale.
        scale_factor = self.heatmap_size / self.img_size
        heatmaps = self._generate_heatmaps(transformed_keypoints, scale_factor)
        
        return image, heatmaps

    def _generate_heatmaps(self, keypoints, scale_factor):
        """
        Generates 4 heatmaps, one for each corner.
        keypoints: List of 4 tuples (x, y) corresponding to TL, TR, BR, BL
        """
        heatmaps = np.zeros((4, self.heatmap_size, self.heatmap_size), dtype=np.float32)
        
        for i, (x, y) in enumerate(keypoints):
            # Scale coordinates to heatmap size
            hx = int(x * scale_factor)
            hy = int(y * scale_factor)
            
            # Generate 2D Gaussian blob
            heatmaps[i] = self._draw_gaussian(heatmaps[i], [hx, hy], self.sigma)
            
        return torch.from_numpy(heatmaps)
        
    def _draw_gaussian(self, heatmap, center, sigma):
        """Draws a 2D Gaussian blob on the heatmap at the center point."""
        tmp_size = sigma * 3
        mu_x = int(center[0] + 0.5)
        mu_y = int(center[1] + 0.5)
        w, h = heatmap.shape[1], heatmap.shape[0]
        
        ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
        br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
        
        if ul[0] >= w or ul[1] >= h or br[0] < 0 or br[1] < 0:
            # Center is outside the image bounds
            return heatmap
            
        size = 2 * tmp_size + 1
        x = np.arange(0, size, 1, np.float32)
        y = x[:, np.newaxis]
        x0 = y0 = size // 2
        
        g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
        
        g_x = max(0, -ul[0]), min(br[0], w) - ul[0]
        g_y = max(0, -ul[1]), min(br[1], h) - ul[1]
        img_x = max(0, ul[0]), min(br[0], w)
        img_y = max(0, ul[1]), min(br[1], h)
        
        heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]] = np.maximum(
            heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]],
            g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
        )
        return heatmap

if __name__ == '__main__':
    # Quick test
    dataset = DocumentCornerDataset('dataset/labels.json', 'dataset/images')
    print(f"Dataset size: {len(dataset)}")
    img, heatmaps = dataset[0]
    print(f"Image shape: {img.shape}")
    print(f"Heatmaps shape: {heatmaps.shape}")
    print(f"Max heatmap values: {[heatmaps[i].max().item() for i in range(4)]}")

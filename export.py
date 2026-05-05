import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import torch
from model import DocumentCornerNet

def export_to_onnx(model_path, output_onnx_path, img_size=256):
    device = torch.device('cpu')
    
    # Initialize model
    model = DocumentCornerNet(pretrained=False)
    
    # Try to load weights if they exist, otherwise export an untrained model structure
    try:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        print(f"Loaded weights from {model_path}")
    except FileNotFoundError:
        print(f"Warning: {model_path} not found. Exporting untrained model architecture.")
        
    model.eval()
    
    # Create dummy input matching the expected shape
    # Batch size 1, 3 channels (RGB), img_size x img_size
    dummy_input = torch.randn(1, 3, img_size, img_size, device=device)
    
    # Export to ONNX
    print(f"Exporting to {output_onnx_path}...")
    torch.onnx.export(
        model, 
        dummy_input, 
        output_onnx_path,
        export_params=True,
        opset_version=18,
        input_names=['input_image'],
        output_names=['heatmaps']
    )
    
    print("ONNX export complete. This model can now be used with ONNXRuntime or TFLite (after conversion).")

if __name__ == '__main__':
    export_to_onnx('weights/best_model.pth', 'doc_corner_net.onnx')

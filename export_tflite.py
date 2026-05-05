import os
import subprocess
import shutil
import sys
import shutil

def convert_onnx_to_tflite(onnx_path, output_tflite_name):
    print(f"Converting {onnx_path} to TFLite using onnx2tf...")
    
    output_dir = "tflite_model_dir"
    os.makedirs(output_dir, exist_ok=True)
    
    # The onnx2tf CLI command
    # -i: Input ONNX model
    # -o: Output directory
    # -cotof: Convert directly to TFLite
    cmd = [
        sys.executable,
        "-m", "onnx2tf",
        "-i", onnx_path,
        "-o", output_dir,
        "-cotof" 
    ]
    
    try:
        # Run the conversion
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        
        # onnx2tf outputs the model named after the input model name
        base_name = os.path.splitext(os.path.basename(onnx_path))[0]
        generated_tflite = os.path.join(output_dir, f"{base_name}_float32.tflite")
        if os.path.exists(generated_tflite):
            shutil.copy(generated_tflite, output_tflite_name)
            print(f"\n[SUCCESS] Conversion successful! TFLite model saved as: {output_tflite_name}")
        else:
            print(f"\nConversion completed but couldn't find the expected float32 tflite file in {output_dir}")
            
    except subprocess.CalledProcessError as e:
        print("\n[ERROR] Error during conversion:")
        print(e.stderr)
        print("\nMake sure you have installed onnx2tf and tensorflow:")
        print("pip install onnx2tf tensorflow")

if __name__ == '__main__':
    convert_onnx_to_tflite('doc_corner_net.onnx', 'doc_corner_net.tflite')

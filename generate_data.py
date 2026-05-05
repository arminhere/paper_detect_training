import os
import cv2
import numpy as np
import random
import urllib.request
import json
from PIL import Image, ImageDraw, ImageFont
from faker import Faker
from tqdm import tqdm

fake = Faker()

def download_backgrounds(num_images, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    existing = len(os.listdir(save_dir))
    if existing >= num_images:
        return
    
    print(f"Downloading {num_images - existing} backgrounds...")
    for i in tqdm(range(existing, num_images)):
        # Using picsum for random background images
        url = f"https://picsum.photos/800/800?random={i}"
        try:
            urllib.request.urlretrieve(url, os.path.join(save_dir, f"bg_{i}.jpg"))
        except Exception as e:
            print(f"Failed to download {url}: {e}")

def generate_fake_document():
    # Random document size (e.g., A4 ratio)
    width = random.randint(500, 800)
    height = int(width * 1.414)
    
    img = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(img)
    
    # Try to load a font, otherwise use default
    try:
        # Just use a basic PIL font if nothing else is available
        font = ImageFont.load_default()
    except:
        font = None
    
    # Draw some random text blocks
    for _ in range(random.randint(5, 15)):
        x = random.randint(20, width // 2)
        y = random.randint(20, height - 50)
        text = fake.paragraph(nb_sentences=random.randint(1, 5))
        draw.text((x, y), text, fill=(0, 0, 0), font=font)
        
    # Draw some random lines
    for _ in range(random.randint(0, 5)):
        x1, y1 = random.randint(20, width), random.randint(20, height)
        x2, y2 = random.randint(20, width), random.randint(20, height)
        draw.line([(x1, y1), (x2, y2)], fill=(0, 0, 0), width=random.randint(1, 3))
        
    return np.array(img)

def generate_synthetic_image(bg_img, doc_img):
    bg_h, bg_w = bg_img.shape[:2]
    doc_h, doc_w = doc_img.shape[:2]
    
    # Document corners: Top-Left, Top-Right, Bottom-Right, Bottom-Left
    src_points = np.array([
        [0, 0],
        [doc_w - 1, 0],
        [doc_w - 1, doc_h - 1],
        [0, doc_h - 1]
    ], dtype=np.float32)
    
    # Generate random destination points within the background
    # Ensure it's somewhat centered but heavily distorted
    margin = int(min(bg_w, bg_h) * 0.1)
    
    tl = [random.randint(margin, bg_w//2), random.randint(margin, bg_h//2)]
    tr = [random.randint(bg_w//2, bg_w-margin), random.randint(margin, bg_h//2)]
    br = [random.randint(bg_w//2, bg_w-margin), random.randint(bg_h//2, bg_h-margin)]
    bl = [random.randint(margin, bg_w//2), random.randint(bg_h//2, bg_h-margin)]
    
    # Add random jitter to simulate steep angles
    dst_points = np.array([tl, tr, br, bl], dtype=np.float32)
    
    # Calculate Homography
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    
    # Warp document
    warped_doc = cv2.warpPerspective(doc_img, M, (bg_w, bg_h))
    
    # Create mask
    mask = np.zeros((doc_h, doc_w), dtype=np.uint8)
    mask.fill(255)
    warped_mask = cv2.warpPerspective(mask, M, (bg_w, bg_h))
    
    # Blend with background
    inverse_mask = cv2.bitwise_not(warped_mask)
    bg_bg = cv2.bitwise_and(bg_img, bg_img, mask=inverse_mask)
    fg_fg = cv2.bitwise_and(warped_doc, warped_doc, mask=warped_mask)
    
    final_img = cv2.add(bg_bg, fg_fg)
    
    # Optional: Add some noise or blur to simulate mobile camera
    if random.random() < 0.5:
        final_img = cv2.GaussianBlur(final_img, (5, 5), random.uniform(0.5, 2.0))
    if random.random() < 0.3:
        noise = np.random.normal(0, random.randint(5, 15), final_img.shape).astype(np.uint8)
        final_img = cv2.add(final_img, noise)
        
    return final_img, dst_points

def main():
    bg_dir = "backgrounds"
    out_dir = "dataset/images"
    os.makedirs(out_dir, exist_ok=True)
    
    num_samples = 1000 # Start with 1000 for quick testing
    download_backgrounds(100, bg_dir)
    
    bg_files = [os.path.join(bg_dir, f) for f in os.listdir(bg_dir) if f.endswith('.jpg')]
    
    labels = []
    
    print("Generating synthetic data...")
    for i in tqdm(range(num_samples)):
        bg_path = random.choice(bg_files)
        bg_img = cv2.imread(bg_path)
        if bg_img is None: continue
        
        # Ensure RGB
        bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB)
        
        # Generate doc
        doc_img = generate_fake_document()
        
        # Transform and blend
        final_img, corners = generate_synthetic_image(bg_img, doc_img)
        
        # Convert back to BGR for saving
        final_img_bgr = cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR)
        
        img_name = f"synth_{i:05d}.jpg"
        cv2.imwrite(os.path.join(out_dir, img_name), final_img_bgr)
        
        labels.append({
            "image": img_name,
            "corners": corners.tolist()
        })
        
    with open("dataset/labels.json", "w") as f:
        json.dump(labels, f, indent=4)
        
    print(f"Generated {num_samples} samples in {out_dir}")

if __name__ == "__main__":
    main()

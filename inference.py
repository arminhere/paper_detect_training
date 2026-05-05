import cv2
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from model import DocumentCornerNet

def extract_coords_from_heatmaps(heatmaps, img_w, img_h):
    """
    Extracts the (x,y) coordinates of the maximum value from each heatmap
    and scales them up to the original image dimensions.
    """
    # heatmaps: [4, H_out, W_out]
    coords = []
    
    # We use sub-pixel approximation for better accuracy
    # Just argmax for now
    for i in range(4):
        hm = heatmaps[i]
        _, max_val, _, max_loc = cv2.minMaxLoc(hm)
        
        # max_loc is (x, y) on the heatmap scale
        x, y = max_loc
        
        # Scale to original image
        hm_h, hm_w = hm.shape
        scale_x = img_w / hm_w
        scale_y = img_h / hm_h
        
        orig_x = int(x * scale_x)
        orig_y = int(y * scale_y)
        
        coords.append([orig_x, orig_y])
        
    return np.array(coords, dtype=np.float32)

def order_points(pts):
    """Order points to Top-Left, Top-Right, Bottom-Right, Bottom-Left."""
    rect = np.zeros((4, 2), dtype="float32")
    
    # The network is trained to output them in order, but we can verify
    # TL will have the smallest sum, BR will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    # TR will have the smallest difference, BL will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    return rect

def four_point_transform(image, pts):
    """Applies perspective warp to crop and flatten the document."""
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    
    # compute the width of the new image
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    # compute the height of the new image
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    # new coords
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")
    
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    
    return warped

def run_inference(image_path, model_path, img_size=256):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = DocumentCornerNet(pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Could not read image")
        
    orig_h, orig_w = image.shape[:2]
    
    transform = A.Compose([
        A.Resize(height=img_size, width=img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_tensor = transform(image=img_rgb)['image'].unsqueeze(0).to(device)
    
    with torch.no_grad():
        heatmaps = model(input_tensor)
        heatmaps = heatmaps.squeeze(0).cpu().numpy()
        
    coords = extract_coords_from_heatmaps(heatmaps, orig_w, orig_h)
    
    warped_doc = four_point_transform(image, coords)
    
    # Draw original image with corners
    img_with_corners = image.copy()
    for pt in coords:
        cv2.circle(img_with_corners, (int(pt[0]), int(pt[1])), 5, (0, 255, 0), -1)
        
    cv2.polylines(img_with_corners, [coords.astype(np.int32)], True, (0, 0, 255), 2)
    
    return img_with_corners, warped_doc

if __name__ == '__main__':
    # Test block (Needs a trained model)
    import sys
    if len(sys.argv) > 2:
        img_p = sys.argv[1]
        model_p = sys.argv[2]
        res_img, res_doc = run_inference(img_p, model_p)
        cv2.imwrite('result_corners.jpg', res_img)
        cv2.imwrite('result_warped.jpg', res_doc)
        print("Inference completed. Saved result_corners.jpg and result_warped.jpg")
    else:
        print("Usage: python inference.py <image_path> <model_path>")

import numpy as np
import cv2
import matplotlib.pyplot as plt

def visualize_mask(mask_path, mask_idx=0):
    """
    Load and visualize a specific mask from a .npy file and save the visualization
    Args:
        mask_path: Path to the .npy file containing masks
        mask_idx: Index of the mask to visualize (default 0)
    """
    # Load masks
    masks = np.load(mask_path)
    print(f"Loaded masks shape: {masks.shape}")
    
    # Get specific mask
    mask = masks[mask_idx]
    print(f"Selected mask shape: {mask.shape}")
    print(f"Mask values range: min={mask.min()}, max={mask.max()}")
    print(f"Unique values in mask: {np.unique(mask)}")
    
    # Create figure with subplots
    plt.figure(figsize=(15, 5))
    
    # Plot original mask
    plt.subplot(131)
    plt.imshow(mask, cmap='gray')
    plt.title('Original Mask')
    plt.colorbar()
    
    # Plot binary mask (threshold if not already binary)
    plt.subplot(132)
    binary_mask = mask.astype(np.uint8) * 255
    plt.imshow(binary_mask, cmap='gray')
    plt.title('Binary Mask')
    plt.colorbar()
    
    # Plot mask overlay on white background
    plt.subplot(133)
    white_bg = np.ones_like(binary_mask) * 255
    overlay = cv2.addWeighted(white_bg.astype(np.uint8), 0.7, binary_mask, 0.3, 0)
    plt.imshow(overlay, cmap='gray')
    plt.title('Mask Overlay')
    plt.colorbar()
    
    plt.tight_layout()
    plt.savefig('mask_visualization.png')
    plt.close()
    print("Saved visualization as 'mask_visualization.png'")

# Example usage:
mask_path = "/ccn2/dataset/babyview/outputs_20250312/yoloe/cdi_10k/2023_1_887640b632_processed_00388_mask.npy"  # Replace with actual path from saved_mask_path column
visualize_mask(mask_path)
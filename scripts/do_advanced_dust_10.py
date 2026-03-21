import os
import cv2
import random
import numpy as np

def generate_perlin_noise_2d(shape, res):
    def f(t):
        return 6*t**5 - 15*t**4 + 10*t**3

    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0:res[0]:delta[0],0:res[1]:delta[1]].transpose(1, 2, 0) % 1
    # Gradients
    angles = 2*np.pi*np.random.rand(res[0]+1, res[1]+1)
    gradients = np.dstack((np.cos(angles), np.sin(angles)))
    g00 = gradients[0:-1,0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g10 = gradients[1:,0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g01 = gradients[0:-1,1:].repeat(d[0], 0).repeat(d[1], 1)
    g11 = gradients[1:,1:].repeat(d[0], 0).repeat(d[1], 1)
    # Ramps
    n00 = np.sum(grid * g00, 2)
    n10 = np.sum(np.dstack((grid[:,:,0]-1, grid[:,:,1])) * g10, 2)
    n01 = np.sum(np.dstack((grid[:,:,0], grid[:,:,1]-1)) * g01, 2)
    n11 = np.sum(np.dstack((grid[:,:,0]-1, grid[:,:,1]-1)) * g11, 2)
    # Interpolation
    t = f(grid)
    n0 = n00*(1-t[:,:,0]) + t[:,:,0]*n10
    n1 = n01*(1-t[:,:,0]) + t[:,:,0]*n11
    return np.sqrt(2)*((1-t[:,:,1])*n0 + t[:,:,1]*n1)

def apply_advanced_dust(image_path, output_path):
    img = cv2.imread(image_path)
    if img is None:
        return False
        
    h, w = img.shape[:2]
    
    # Generate varied realistic dust cloud noise
    noise1 = generate_perlin_noise_2d((h, w), (4, 4))
    noise2 = generate_perlin_noise_2d((h, w), (8, 8))
    
    # Combine noises for a more wispy, natural look
    dust_mask = (noise1 + noise2 * 0.5)
    dust_mask = (dust_mask - dust_mask.min()) / (dust_mask.max() - dust_mask.min() + 1e-5)
    
    # Create the color of the dust (sandy, yellowish-brown haze)
    dust_color = np.zeros_like(img, dtype=np.float32)
    dust_color[:] = [150, 180, 200]  # BGR format (light yellowish-brown/grey)
    
    # Add a depth gradient (usually more dust far away / higher up in the air)
    gradient = np.linspace(1.2, 0.4, h).reshape(-1, 1, 1)
    
    # Create final dust blend map
    blend_map = (dust_mask[:, :, np.newaxis] * 0.6 + 0.4) * gradient
    blend_map = np.clip(blend_map, 0, 0.85)  # Don't completely white-out the image
    
    # Blend image and dust, then reduce overall contrast
    img_float = img.astype(np.float32)
    blended = img_float * (1 - blend_map) + dust_color * blend_map
    
    # Reduce contrast and add a subtle sepia/dusty tone across the whole image
    blended = (blended - 128) * 0.8 + 128
    
    # A bit of uniform high-frequency noise (grainy dirt in the air)
    grain = np.random.normal(0, 5, (h, w, 3)).astype(np.float32)
    blended += grain
    
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    # Add a slight blur to simulate hazy particles
    blended = cv2.GaussianBlur(blended, (3, 3), 0)
    
    cv2.imwrite(output_path, blended)
    return True

if __name__ == "__main__":
    src_folder = r"E:\Company\Green Build AI\Prototypes\BuildSight\Dataset\Indian Dataset\Normal_Site_Condition"
    dst_folder = r"E:\Company\Green Build AI\Prototypes\BuildSight\Dataset\Indian Dataset\Dusty_Condition"

    files = [f for f in os.listdir(src_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    random.shuffle(files)

    generated = 0
    for f in files:
        src_path = os.path.join(src_folder, f)
        dst_path = os.path.join(dst_folder, f"advanced_dust_aug_V2_{generated}_{f}")
        if apply_advanced_dust(src_path, dst_path):
            print(f"Generated {dst_path}")
            generated += 1
        if generated >= 10:
            break

    print(f"Successfully generated {generated} advanced dust augmented images.")

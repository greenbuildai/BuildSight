import os
import glob
import random
from PIL import Image, ImageDraw, ImageFont

def create_collage():
    base_dir = r"D:\Jovi\Projects\BuildSight\Dataset\Indian Dataset"
    conditions = ['Normal_Site_Condition', 'Dusty_Condition', 'Low_Light_Condition', 'Crowded_Condition']
    
    # Grid settings
    img_width, img_height = 800, 600
    rows, cols = 4, 4
    header_height = 80
    bg_color = (25, 25, 25)
    text_color = (255, 255, 255)
    padding = 20
    
    # Calculate total image size
    total_width = (img_width * cols) + (padding * (cols + 1))
    total_height = (img_height * rows) + (header_height * rows) + (padding * (rows + 1))
    
    collage = Image.new('RGB', (total_width, total_height), bg_color)
    draw = ImageDraw.Draw(collage)
    
    try:
        font = ImageFont.truetype("arialbd.ttf", 60)
    except IOError:
        font = ImageFont.load_default()
    
    y_offset = padding
    
    # Specific images attached by user
    img1_concrete = r"C:\Users\brigh\.gemini\antigravity\brain\bb385151-baf7-4b8f-888e-4df569c91f90\media__1776246021515.jpg"
    img2_person = r"C:\Users\brigh\.gemini\antigravity\brain\bb385151-baf7-4b8f-888e-4df569c91f90\media__1776246103274.jpg"
    
    # Set seed
    random.seed(42)
    
    for row, condition in enumerate(conditions):
        path = os.path.join(base_dir, condition)
        files = glob.glob(os.path.join(path, '*.*'))
        img_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # Pick 4 random or specific images
        if len(img_files) >= 4:
            selected_files = random.sample(img_files, 4)
        else:
            selected_files = img_files
            
        # Apply specific replacements
        if condition == 'Normal_Site_Condition':
            # Replace 1st image from normal site condition with the provided 2nd image
            selected_files[0] = img2_person
        elif condition == 'Crowded_Condition':
            # Replace 2nd image in the crowded condition with the 1st image attached
            selected_files[1] = img1_concrete
            
        # Draw condition title
        title = condition.replace('_', ' ')
        draw.text((padding, y_offset + 10), title, font=font, fill=text_color)
        
        row_y = y_offset + header_height
        
        for col, img_path in enumerate(selected_files):
            try:
                img = Image.open(img_path)
                
                # Crop and resize exactly to aspect ratio
                # Center crop to maintain aspect ratio without distortion
                img_aspect = img.width / img.height
                target_aspect = img_width / img_height
                
                if img_aspect > target_aspect:
                    # image is wider than target
                    new_width = int(img.height * target_aspect)
                    left = (img.width - new_width) / 2
                    img = img.crop((left, 0, left + new_width, img.height))
                else:
                    # image is taller than target
                    new_height = int(img.width / target_aspect)
                    top = (img.height - new_height) / 2
                    img = img.crop((0, top, img.width, top + new_height))
                    
                img = img.resize((img_width, img_height), Image.Resampling.LANCZOS)
                
                x_offset = padding + col * (img_width + padding)
                collage.paste(img, (x_offset, row_y))
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
        
        y_offset += header_height + img_height + padding
        
    output_path = r"e:\Company\Green Build AI\Prototypes\BuildSight\site_conditions_collage.jpg"
    collage.save(output_path, quality=95)
    print(f"Collage updated successfully and saved to {output_path}")

if __name__ == '__main__':
    create_collage()

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
        # Try to load a nice font, fallback to default
        font = ImageFont.truetype("arialbd.ttf", 60)
    except IOError:
        font = ImageFont.load_default()
    
    y_offset = padding
    
    # Set seed for reproducible "best-looking" random selection
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
    print(f"Collage saved successfully to {output_path}")

if __name__ == '__main__':
    create_collage()

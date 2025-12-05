import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import random
import string
from pathlib import Path
import os


class GeneratorConfig:

    OUTPUT_DIR = "train" # or test or any other name
    NUM_IMAGES = 5000
    
    IMG_WIDTH = 700   
    IMG_HEIGHT = 200  
    
    LENGTH = 5
    CHARACTERS = string.ascii_uppercase + string.digits
    
    FONT_PATH = None  # use system fonts
    FONT_SIZE_MIN = 80  
    FONT_SIZE_MAX = 110
    
    COLORS = [
        # (background_color, text_color, pattern_color)
        ((255, 210, 180), (220, 150, 120), (255, 230, 200)),  # Peach
        ((180, 220, 210), (120, 180, 170), (200, 235, 225)),  # Teal 
        ((200, 240, 200), (140, 200, 140), (220, 250, 220)),  # Light green 
        ((240, 200, 230), (210, 150, 200), (250, 220, 240)),  # Pink
    ]
    

    NOISE_DENSITY = 0.012
    NOISE_DOT_SIZE_MIN = 2
    NOISE_DOT_SIZE_MAX = 4
    

    BORDER_RADIUS = 80
    BORDER_THICKNESS = 3
    BORDER_COLOR = (80, 80, 80)

config = GeneratorConfig()


def create_rounded_rectangle_mask(width, height, radius):

    mask = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle(
        [(0, 0), (width, height)],
        radius=radius,
        fill=255
    )
    return mask


def add_rectangular_patterns(img, bg_color, pattern_color):

    img = img.convert('RGBA')
    overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    
    width, height = img.size
    

    num_blocks = random.randint(3, 6)
    for _ in range(num_blocks):

        block_width = random.randint(width//3, width//2)
        block_height = random.randint(height//2, int(height * 0.8))
        
        x = random.randint(-block_width//2, width - block_width//2)
        y = random.randint(-block_height//2, height - block_height//2)
        

        pattern_rgba = (*pattern_color, random.randint(60, 100))
        
        draw.rectangle([x, y, x+block_width, y+block_height], fill=pattern_rgba)
    

    num_circles = random.randint(2, 4)
    for _ in range(num_circles):
        diameter = random.randint(width//3, int(width * 0.6))
        x = random.randint(-diameter//2, width - diameter//2)
        y = random.randint(-diameter//2, height - diameter//2)
        
        pattern_rgba = (*pattern_color, random.randint(50, 90))
        draw.ellipse([x, y, x+diameter, y+diameter], fill=pattern_rgba)
    

    result = Image.alpha_composite(img, overlay)
    return result.convert('RGB')


def add_noise_dots(img, density=0.012):

    img_array = np.array(img)
    height, width = img_array.shape[:2]
    
    num_dots = int(width * height * density)
    
    for _ in range(num_dots):
        x = random.randint(0, width - 1)
        y = random.randint(0, height - 1)
        dot_size = random.randint(config.NOISE_DOT_SIZE_MIN, config.NOISE_DOT_SIZE_MAX)
        
        # Black dots
        cv2.circle(img_array, (x, y), dot_size, (0, 0, 0), -1)
    
    return Image.fromarray(img_array)


def get_random_text(length=None):

    if length is None:
        length = config.LENGTH
    return ''.join(random.choices(config.CHARACTERS, k=length))


def try_load_bold_font(size):

    font_paths = [
        # Windows
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/ariblk.ttf",
        "C:/Windows/Fonts/impact.ttf",
        "C:/Windows/Fonts/BROADW.TTF",
        # Linux
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        # Mac
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial Bold.ttf",
    ]
    
    if config.FONT_PATH and os.path.exists(config.FONT_PATH):
        try:
            return ImageFont.truetype(config.FONT_PATH, size)
        except:
            pass
    
    for path in font_paths:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except:
                continue
    
    return ImageFont.load_default()


def draw_bold_text(img, text, font, text_color):

    draw = ImageDraw.Draw(img)
    width, height = img.size
    

    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # Center the text
    x = (width - text_width) // 2
    y = (height - text_height) // 2
    

    offsets = [
        (0, 0),
        (-1, 0), (1, 0), (0, -1), (0, 1),
        (-1, -1), (1, -1), (-1, 1), (1, 1),
        (-2, 0), (2, 0), (0, -2), (0, 2),
    ]
    
    for offset_x, offset_y in offsets:
        draw.text((x + offset_x, y + offset_y), text, font=font, fill=text_color)
    
    return img


def add_subtle_transparency_to_text(img, text, font, bg_color):
    return img


def generate_captcha_image(text=None, save_path=None):

    if text is None:
        text = get_random_text()
    

    bg_color, text_color, pattern_color = random.choice(config.COLORS)
    
    # Create base image with background color
    img = Image.new('RGB', (config.IMG_WIDTH, config.IMG_HEIGHT), bg_color)
    
    # Add rectangular patterns BEFORE text (background layer)
    img = add_rectangular_patterns(img, bg_color, pattern_color)
    
    # Create rounded rectangle mask
    mask = create_rounded_rectangle_mask(
        config.IMG_WIDTH,
        config.IMG_HEIGHT,
        config.BORDER_RADIUS
    )
    
    # Apply rounded corners
    img.putalpha(mask)
    background = Image.new('RGB', (config.IMG_WIDTH, config.IMG_HEIGHT), (255, 255, 255))
    background.paste(img, (0, 0), img)
    img = background
    
    # Add border
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle(
        [(0, 0), (config.IMG_WIDTH-1, config.IMG_HEIGHT-1)],
        radius=config.BORDER_RADIUS,
        outline=config.BORDER_COLOR,
        width=config.BORDER_THICKNESS
    )
    

    font_size = random.randint(config.FONT_SIZE_MIN, config.FONT_SIZE_MAX)
    font = try_load_bold_font(font_size)
    

    img = draw_bold_text(img, text, font, text_color)
    

    img = add_noise_dots(img, density=config.NOISE_DENSITY)
    

    if save_path:
        img.save(save_path)
    
    return img, text


def generate_dataset(num_images=5000, output_dir="train"):

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for i in range(num_images):
        text = get_random_text()
        filename = f"{text}_{i:05d}.png"
        save_path = output_path / filename
        
        generate_captcha_image(text, save_path)
        
        if (i + 1) % 100 == 0:
            print(f"Generated {i + 1}/{num_images} images")
    
    print("-" * 60)
    print(f"Successfully generated {num_images} images!")
    print(f"Location: {output_path.absolute()}")


def generate_sample_preview(num_samples=10):

    import matplotlib.pyplot as plt
        
    fig, axes = plt.subplots(2, 5, figsize=(25, 10))
    axes = axes.flatten()
    
    for idx in range(num_samples):
        img, text = generate_captcha_image()
        axes[idx].imshow(img)
        axes[idx].set_title(f"Text: {text}", fontsize=14, weight='bold')
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig('captcha_samples_preview.png', dpi=150, bbox_inches='tight')
    print("Preview saved as 'captcha_samples_preview.png'")
    plt.show()


if __name__ == "__main__":
    
    #generate_sample_preview(num_samples=10)
    
    generate_dataset(
        num_images=config.NUM_IMAGES,
        output_dir=config.OUTPUT_DIR
    )





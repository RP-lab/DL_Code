import os
import random
from PIL import Image, ImageDraw
import numpy as np


def generate_image(save_path, label, image_size=(128, 128), count=100):
    os.makedirs(save_path, exist_ok=True)
    for i in range(count):
        # Set a unique seed for each image to ensure diversity
        random.seed(i)
        np.random.seed(i)

        img = Image.new("RGB", image_size)
        draw = ImageDraw.Draw(img)

        if label == "defective":
            # Generate base random noise
            noise = np.random.randint(0, 255, (image_size[1], image_size[0], 3), dtype=np.uint8)
            img = Image.fromarray(noise)

            # Add random defects like shapes
            for _ in range(random.randint(5, 15)):  # Number of defects
                shape_type = random.choice(['line', 'rectangle', 'ellipse'])
                x1, y1 = random.randint(0, image_size[0] - 1), random.randint(0, image_size[1] - 1)
                x2, y2 = random.randint(x1, image_size[0]), random.randint(y1, image_size[1])
                color = tuple(np.random.randint(0, 255, 3))

                if shape_type == 'line':
                    draw.line((x1, y1, x2, y2), fill=color, width=random.randint(1, 3))
                elif shape_type == 'rectangle':
                    draw.rectangle((x1, y1, x2, y2), outline=color, width=random.randint(1, 3))
                elif shape_type == 'ellipse':
                    draw.ellipse((x1, y1, x2, y2), outline=color, width=random.randint(1, 3))

        else:
            # Generate a smooth gradient for non-defective images
            for x in range(image_size[0]):
                for y in range(image_size[1]):
                    color = int((x + y + random.randint(0, 20)) % 256)  # Add randomness
                    draw.point((x, y), fill=(color, color, color))

        # Save the generated image
        img.save(os.path.join(save_path, f"{label}_{i}.png"))


def generate_dataset(base_path, split_ratios=(0.7, 0.2, 0.1), image_count=300):
    os.makedirs(base_path, exist_ok=True)
    splits = ['train', 'val', 'test']
    labels = ['defective', 'non_defective']

    for split, ratio in zip(splits, split_ratios):
        split_path = os.path.join(base_path, split)
        for label in labels:
            label_path = os.path.join(split_path, label)
            generate_image(label_path, label, count=int(image_count * ratio))


if __name__ == "__main__":
    base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data")
    generate_dataset(base_path=base_path, image_count=300)

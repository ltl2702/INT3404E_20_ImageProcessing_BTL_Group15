import os 
import cv2 
import numpy as np 
from PIL import Image  
from torchvision import transforms  
from random import sample  

# Path to the directory containing training images
root_dir = 'wb_recognition_dataset/train/images'

# Hàm trả về đối tượng biến đổi ảnh
def get_augmentation():
    """
    Returns a transform object with random augmentations.
    The augmentations include:
    - Random rotation within the range (-15, 15) degrees
    - Random adjustments of brightness, contrast, saturation, and hue
    - Random conversion to grayscale with a probability of 0.2
    """
    transform = transforms.Compose([
        transforms.RandomRotation(degrees=15), 
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.RandomGrayscale(p=0.2)  
    ])
    return transform

# Maximum number of images required in each folder
max_images_per_folder = 30  

# Iterate through each subdirectory in the root directory
for dir_name in os.listdir(root_dir):
    dir_path = os.path.join(root_dir, dir_name)
    
    if os.path.isdir(dir_path):
        # Get the list of images in the directory and count the current number of images
        images = os.listdir(dir_path) 
        num_images = len(images)  
        
        if num_images < max_images_per_folder:
            i = 1
            # Add new images until the required number is reached
            while num_images < max_images_per_folder:
                # Randomly select an image from the existing images
                image_name = sample(images, k=1)[0]
                image_path = os.path.join(dir_path, image_name) 
                image = cv2.imread(image_path)  

                # Apply image augmentation
                aug = get_augmentation()
                augmented_image = aug(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))

                # Create and save the augmented image with a new name
                new_image_name = f'{image_name.split(".")[0]}_augmented_{i}.png'
                new_image_path = os.path.join(dir_path, new_image_name)
                augmented_image.save(new_image_path)  

                num_images += 1 
                i += 1  

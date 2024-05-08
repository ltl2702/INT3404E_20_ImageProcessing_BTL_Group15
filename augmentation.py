import os
import cv2
import numpy as np
from albumentations import Compose, ColorJitter, ToGray, Rotate
from random import sample

root_dir = 'wb_recognition_dataset/train/images'

#random random
def get_random_augmentation_parameters():
    brightness = np.random.uniform(0.3, 0.45)
    contrast = np.random.uniform(0.1, 0.5)
    saturation = np.random.uniform(0.8, 1.2)
    hue = np.random.uniform(0, 0.3)
    to_gray_prob = np.random.uniform(0.5, 1.0)
    return brightness, contrast, saturation, hue, to_gray_prob

def get_augmentation_pipeline():
    brightness, contrast, saturation, hue, to_gray_prob = get_random_augmentation_parameters()
    rotation_angle = np.random.uniform(-20, 20)
    aug = Compose([
        ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue, always_apply=True),
        ToGray(p=to_gray_prob),
        Rotate(limit=rotation_angle, interpolation=cv2.INTER_LINEAR, p=1.0)
    ])
    return aug

max_images_per_folder = 30

# duyệt từng thư mục con
for dir_name in os.listdir(root_dir):
    dir_path = os.path.join(root_dir, dir_name)
    
    # đường dẫn có phải là thư mục không
    if os.path.isdir(dir_path):
        images = os.listdir(dir_path)
        num_images = len(images)
        
        # ảnh nhỏ hơn 30, tăng cường dữ liệu
        if num_images < max_images_per_folder:
            i = 1
            while num_images < max_images_per_folder:
                image_name = sample(images, k=1)[0]
                image_path = os.path.join(dir_path, image_name)
                image = cv2.imread(image_path)

                aug = get_augmentation_pipeline()
                augmented = aug(image=image)

                # Lưu
                new_image_name = f'{image_name.split(".")[0]}_augmented_{i}.png'
                new_image_path = os.path.join(dir_path, new_image_name)
                cv2.imwrite(new_image_path, augmented['image'])

                num_images += 1
                i += 1

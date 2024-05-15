import os
import cv2
import numpy as np
from albumentations import Compose, ColorJitter, ToGray, Rotate
from random import sample

def calculate_image_counts(folder_path):
    image_counts = {}
    for folder_name in os.listdir(folder_path):
        folder_dir = os.path.join(folder_path, folder_name)
        if os.path.isdir(folder_dir):
            num_images = len(os.listdir(folder_dir))
            image_counts[folder_name] = num_images
    return image_counts

def normalize_image_counts(image_counts):
    mean_count = np.mean(list(image_counts.values()))
    std_dev = np.std(list(image_counts.values()))
    return mean_count, std_dev

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

def perform_augmentation(root_dir, target_distribution):
    for dir_name in os.listdir(root_dir):
        dir_path = os.path.join(root_dir, dir_name)
        if os.path.isdir(dir_path):
            images = os.listdir(dir_path)
            num_images = len(images)
            target_num_images = target_distribution[dir_name]
            # print((num_images - train_mean_count)/train_std_dev)
            
            if num_images < target_num_images:
                augmentation_ratio = target_num_images // num_images


                for i in range(augmentation_ratio):
                    image_name = sample(images, k=1)[0]
                    image_path = os.path.join(dir_path, image_name)
                    image = cv2.imread(image_path)

                    aug = get_augmentation_pipeline()
                    augmented = aug(image=image)

                    new_image_name = f'{image_name.split(".")[0]}_augmented_{i}.png'
                    new_image_path = os.path.join(dir_path, new_image_name)
                    
                    # Kiểm tra xem ảnh mới có trùng với ảnh nào trong thư mục không
                    while any(cv2.imread(os.path.join(dir_path, existing_image)).tolist() == augmented['image'].tolist() for existing_image in images):
                        aug = get_augmentation_pipeline()
                        augmented = aug(image=image)
                    cv2.imwrite(new_image_path, augmented['image'])


train_folder_path = 'wb_recognition_dataset/train/images'

# Tính phân bố số lượng ảnh của các thư mục con trong thư mục train
train_image_counts = calculate_image_counts(train_folder_path)

# Trung bình và độ lệch chuẩn số lượng ảnh của các thư mục con trong thư mục train
train_mean_count, train_std_dev = normalize_image_counts(train_image_counts)

# Tính toán phân bố mục tiêu cho tập train dựa trên phân bố chuẩn
target_train_distribution = {}
for folder, count in train_image_counts.items():
    target_train_distribution[folder] = int(np.random.normal(train_mean_count, train_std_dev))

# Thực hiện augmentation để tăng số lượng ảnh trong các thư mục con của tập train
perform_augmentation(train_folder_path, target_distribution=target_train_distribution)

# print("Image count:", train_image_counts)
print("Trung bình: ", train_mean_count)
print("Độ lệch chuẩn: ", train_std_dev)
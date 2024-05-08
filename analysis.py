import os
import numpy as np
import matplotlib.pyplot as plt

folder_path = "wb_recognition_dataset/train/images/"

image_counts = {}

for folder_index in range(2139):
    folder_name = str(folder_index) 
    folder_dir = os.path.join(folder_path, folder_name)
    if os.path.isdir(folder_dir):
        num_images = len(os.listdir(folder_dir))
        image_counts[folder_name] = num_images
    else:
        print(f"Thư mục '{folder_name}' không tồn tại.")

most_images_folder = max(image_counts, key=image_counts.get)
least_images_folder = min(image_counts, key=image_counts.get)

print(f"Folder có nhiều ảnh nhất: {most_images_folder}, số lượng ảnh: {image_counts[most_images_folder]}")
print(f"Folder có ít ảnh nhất: {least_images_folder}, số lượng ảnh: {image_counts[least_images_folder]}")

#giá trị trung bình và độ lệch chuẩn
mean_count = np.mean(list(image_counts.values()))
std_dev = np.std(list(image_counts.values()))

print(f"Giá trị trung bình: {mean_count}")
print(f"Độ lệch chuẩn: {std_dev}")

#khoảng data tập trung chủ yếu
lower_bound = int(mean_count - std_dev)
upper_bound = int(mean_count + std_dev)

print(f"Data tập trung chủ yếu trong khoảng từ {lower_bound} đến {upper_bound} ảnh.")

folders = list(image_counts.keys())
counts = list(image_counts.values())

plt.figure(figsize=(18, 10))
plt.bar(folders, counts, color='skyblue')
plt.xlabel('Tên thư mục')
plt.ylabel('Số lượng ảnh')
plt.title('Số lượng ảnh trong mỗi thư mục')
plt.xticks(rotation=0, ha='right')
plt.axhline(mean_count, color='red', linestyle='--', label='Giá trị trung bình')
plt.axhspan(lower_bound, upper_bound, color='yellow', alpha=0.3, label='Khoảng tập trung chủ yếu')
plt.legend()
plt.tight_layout()
plt.show()

import os
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

def count_images_in_subfolders(folder_path):
    # Tạo một danh sách để lưu trữ số lượng ảnh trong mỗi thư mục con
    image_counts = []
    # Lặp qua tất cả các thư mục con và tìm số lượng ảnh trong mỗi thư mục con
    for subfolder in os.listdir(folder_path):
        subfolder_path = os.path.join(folder_path, subfolder)
        if os.path.isdir(subfolder_path):
            num_images = len(os.listdir(subfolder_path))
            image_counts.append(num_images)

            # Nếu thư mục con có thư mục con khác, tiến hành đệ quy
            if os.listdir(subfolder_path):
                image_counts.extend(count_images_in_subfolders(subfolder_path))

    return image_counts

folder_path = "wb_recognition_dataset/train/images"

# Tạo một danh sách để lưu trữ số lượng ảnh trong mỗi thư mục con
all_image_counts = count_images_in_subfolders(folder_path)

# Sử dụng Counter để đếm tần suất xuất hiện của các số lượng ảnh trong các thư mục con
count_of_counts = Counter(all_image_counts)

# Tìm số lượng ảnh và tần suất xuất hiện nhiều nhất
most_common_count, frequency = count_of_counts.most_common(1)[0]

print(f"Số lượng ảnh xuất hiện nhiều nhất trong mỗi thư mục con: {most_common_count}")
print(f"Tần suất xuất hiện của số lượng ảnh này: {frequency}")

# Vẽ biểu đồ
plt.figure(figsize=(18, 10))
plt.bar(count_of_counts.keys(), count_of_counts.values(), color='skyblue')
plt.xlabel('Số lượng ảnh')
plt.ylabel('Số lượng thư mục')
plt.title('Tần suất số lượng ảnh trong mỗi thư mục con')
plt.axhline(frequency, color='red', linestyle='--', label='Tần suất cao nhất')
plt.legend()
plt.tight_layout()
plt.show()

import os 
import cv2 
import numpy as np 
from PIL import Image  
from torchvision import transforms  
from random import sample  

# Đường dẫn đến các thư mục trong tập train
root_dir = 'wb_recognition_dataset/train/images'

# Hàm trả về đối tượng biến đổi ảnh
def get_augmentation():
    transform = transforms.Compose([
        # Xoay ngẫu nhiên ảnh trong khoảng (-15, 15) độ
        transforms.RandomRotation(degrees=15),  
        # Thay đổi ngẫu nhiên độ sáng, tương phản, bão hòa và hue
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        # Biến đổi ngẫu nhiên ảnh thành ảnh xám với xác suất 0.2
        transforms.RandomGrayscale(p=0.2)  
    ])
    return transform

# Số lượng ảnh tối đa mỗi thư mục cần có
max_images_per_folder = 30  

# Duyệt qua từng thư mục con trong thư mục gốc
for dir_name in os.listdir(root_dir):
    dir_path = os.path.join(root_dir, dir_name)
    
    # Kiểm tra nếu đường dẫn hiện tại là thư mục
    if os.path.isdir(dir_path):
        images = os.listdir(dir_path)  # Lấy danh sách các ảnh trong thư mục
        num_images = len(images)  # Đếm số lượng ảnh hiện có
        
        # Nếu số lượng ảnh ít hơn số lượng tối đa yêu cầu
        if num_images < max_images_per_folder:
            i = 1
            # Tiếp tục thêm ảnh mới cho đến khi đạt đủ số lượng tối đa
            while num_images < max_images_per_folder:
                # Lấy ngẫu nhiên một ảnh từ danh sách các ảnh hiện có
                image_name = sample(images, k=1)[0]
                image_path = os.path.join(dir_path, image_name) #Tạo đường dẫn
                # Đọc ảnh bằng OpenCV
                image = cv2.imread(image_path)  

                # Lấy đối tượng biến đổi ảnh
                aug = get_augmentation()
                # Áp dụng biến đổi ảnh và chuyển đổi định dạng ảnh từ BGR sang RGB
                augmented_image = aug(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))

                # Tạo tên mới cho ảnh đã biến đổi
                new_image_name = f'{image_name.split(".")[0]}_augmented_{i}.png'
                new_image_path = os.path.join(dir_path, new_image_name)
                # Lưu ảnh đã biến đổi vào thư mục
                augmented_image.save(new_image_path)  

                num_images += 1  # Tăng số lượng ảnh lên
                i += 1  # Tăng chỉ số để đặt tên cho ảnh mới

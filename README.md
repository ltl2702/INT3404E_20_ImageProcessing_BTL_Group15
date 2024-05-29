# Image Processing Assignment - Sinonom Character Recognition

## Giới Thiệu

Đây là repo GitHub cho bài tập lớn giữa kì môn Xử lý Ảnh.

- **Mã Môn Học:** INT3404E 20
- **Nhóm:** 15
- **Chủ Đề:** 2 - Nhận dạng ký tự Nôm

## Thành Viên

- Ngô Tùng Lâm MSV: 22028092
- Lương Thị Linh MSV: 22028202
- Nguyễn Văn Sơn MSV: 22028020
- Nguyễn Thị Thu Trang MSV: 22028254

## Mục tiêu 
- Nghiên cứu các phương pháp phân tích và xử lý dữ liệu
- Nghiên cứu các mô hình học sâu trong bài toán phân loại hình ảnh
- Xây dựng, huấn luyện mô hình phân loại các hình ảnh ký tự Nôm

## Hướng dẫn cài đặt và sử dụng


Dưới đây là hướng dẫn để bắt đầu với dự án của chúng tôi:

1. Clone repository này về máy tính của bạn:

- git clone https://github.com/sondopin/INT3404E_20_ImageProcessing_BTL_Group15.git
 
3. Tiến hành cài đặt các công cụ và thư viện cần thiết.

4. Mở các tệp và tài liệu hướng dẫn để bắt đầu làm việc.

## Cấu trúc của repository
***bash
│   README.md
│
├───cnn_model
│       config.py
│       model.pt
│       MyDataset.py
│       OCR_Model.py
│       train.py
│       ultis.py
│
├───data_processing
│       augmentation.py
│       data-analysis.ipynb
│       reformat_dataset.py
│
├───evaluate
│       efficientNetb0-imagenet-05-best(95.47).pt
│       evaluate.ipynb
│       requirements.txt
│       task2.py
│
├───report
│       Full Report.pdf
│       Planning-and-Proposal-Development-report-Group15.pdf
│
└───transfer_learning
    ├───EfficientNetB0
    │   │   EfficientNetB0_transfer_learning.ipynb
    │   │
    │   └───saved_model
    │           efficientNetb0-customed-5-best.pt
    │           efficientNetb0-customed-5-last.pt
    │           efficientNetb0-customed-6-best.pt
    │           efficientNetb0-customed-6-last.pt
    │           efficientNetb0-imagenet-00-best.pt
    │           efficientNetb0-imagenet-00-last.pt
    │           efficientNetb0-imagenet-00.pt
    │           efficientNetb0-imagenet-01.pt
    │           efficientNetb0-imagenet-02.pt
    │           efficientNetb0-imagenet-03.pt
    │           efficientNetb0-imagenet-04-best.pt
    │           efficientNetb0-imagenet-04-last.pt
    │
    ├───Resnet18
    │   │   Resnet18_transfer_learning.ipynb
    │   │
    │   └───saved_model
    │           resnet18-ImageNet-0.pt
    │           resnet18-imagenet-01-best.pt
    │           resnet18-imagenet-01-last.pt
    │           resnet18-imagenet-02-1-best.pt
    │           resnet18-imagenet-02-1-last.pt
    │
    ├───Resnet50
    │   │   resnet50-transfer-learning.ipynb
    │   │
    │   └───saved_model
    │
    ├───vgg16
    │   │   vgg16-transfer-learning.ipynb
    │   │
    │   └───saved_model
    │
    └───yolov8
        │   yolov8-transfer-learning.ipynb
        │
        └───saved_model
                best.pt
                last.pt
***
## Kết quả
Mô hình của nhóm đạt độ chính xác 95,8% trên tập dữ liệu test của thầy

## Ghi Chú

- Hãy tuân thủ quy tắc của trường đối với việc sử dụng mã nguồn và dữ liệu.
- Nếu có bất kỳ vấn đề hoặc câu hỏi, vui lòng liên hệ với một trong các thành viên trong nhóm.

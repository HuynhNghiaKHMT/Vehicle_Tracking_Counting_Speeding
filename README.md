# Yolo + ByteTracker + Counting + Speeding Application (CS338)

![GitHub last commit](https://img.shields.io/github/last-commit/HuynhNghiaKHMT/Vehicle_Tracking_Counting_Speeding
)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📝 Giới thiệu

Dự án này là một ứng dụng theo dõi đa đối tượng (MOT - Multiple Object Tracking) toàn diện, tích hợp các công nghệ tiên tiến như YOLO (You Only Look Once) để phát hiện đối tượng, ByteTrack để theo dõi, cùng với các module đếm số lượng và ước tính tốc độ phương tiện. Dự án được phát triển nhằm mục đích giáo dục trong khuôn khổ môn học CS338 - Pattern Regconition.


## ✨ Tính năng chính

- 🚗 **Phát hiện phương tiện:** Sử dụng YOLOv8, RT-DETR để phát hiện chính xác các phương tiện giao thông trong video.
- 🧭 **Theo dõi đối tượng:** Tích hợp ByteTrack để theo dõi liên tục phương tiện qua nhiều khung hình, xử lý tốt các tình huống che khuất hoặc di chuyển nhanh.
- 🔢 **Đếm phương tiện:** Xác định số lượng phương tiện đi qua một khu vực hoặc đường ranh giới định nghĩa trước.
- ⚡ **Ước tính tốc độ:** Tính toán và hiển thị tốc độ ước lượng của từng phương tiện dựa trên chuyển động và tỷ lệ không gian.
- 💻 **Giao diện người dùng:** Hỗ trợ giao diện Web trực quan sử dụng **Streamlit**.
- 🎓 **Hướng đến giáo dục:** Mã nguồn được tổ chức rõ ràng, thuận tiện cho mục đích học tập và nghiên cứu.

## 📦 Công nghệ và Thư viện sử dụng

- [YOLOv8](https://github.com/ultralytics/yolov8) và [RT-DETR](https://docs.ultralytics.com/vi/models/rtdetr/): Phát hiện đối tượng thời gian thực.
- [ByteTrack](https://github.com/ifzhang/ByteTrack): Thuật toán theo dõi MOT hiệu quả.
- [OpenCV](https://opencv.org/): Xử lý ảnh và video.
- [Streamlit](https://streamlit.io/): Giao diện Web tương tác.
- Python packages: `numpy`, `matplotlib`, `pandas`, `scikit-learn`, v.v.

Dự án sử dụng và tham khảo các repo tracking gốc sau:

* **Sort:**
    * [abewley/sort](https://github.com/abewley/sort)
    * [FoundationVision/ByteTrack/yolox/sort_tracker](https://github.com/FoundationVision/ByteTrack/tree/main/yolox/sort_tracker)
* **DeepSort:**
    * [nwojke/deep_sort](https://github.com/nwojke/deep_sort)
    * [FoundationVision/ByteTrack/yolox/deepsort_tracker](https://github.com/FoundationVision/ByteTrack/tree/main/yolox/deepsort_tracker)
* **ByteTrack:**
    * [FoundationVision/ByteTrack/yolox/tracker](https://github.com/FoundationVision/ByteTrack/tree/main/yolox/tracker)

## 📂 Cấu trúc thư mục
Dưới đây là cấu trúc thư mục chính của dự án, giúp bạn dễ dàng điều hướng và hiểu rõ các thành phần:
```bash
    Vehicle_Tracking_Counting_Speeding/
├── streamlit/                      # Chứa mã nguồn cho ứng dụng giao diện web Streamlit.
├── vscode/                         # Cấu hình và thiết lập dành cho môi trường VS Code.
├── config/                         # Các file cấu hình cho ứng dụng và mô hình.
├── util/                           # Các module và hàm tiện ích chung.
├── annotations/                    # Dữ liệu annotations (nhãn) của video.
│   ├── groundtruth/                # Ground truth annotations dùng để đánh giá.
│   ├── predict/                    # Annotations dự đoán từ mô hình tracking.
│   └── txt/                        # Các file annotation dạng văn bản thô.
├── assets/                         # Tài nguyên tĩnh như ảnh, GIF, video demo.
├── video/                          # Các file video đầu vào.
├── notebook/                       # Các Jupyter Notebook để thử nghiệm và phân tích.
├── detector/                       # Mã nguồn cho mô hình phát hiện đối tượng.
├── model/                          # Các file trọng số (weights) của mô hình đã huấn luyện.
├── datamanager/                    # Module quản lý và tiền xử lý dữ liệu.
├── dataloading/                    # Các lớp và hàm để tải dữ liệu.
├── sort_tracker/                   # Triển khai thuật toán theo dõi SORT.
├── deepsort_tracker/               # Triển khai thuật toán theo dõi DeepSORT.
├── bytetrack_tracker/              # Triển khai thuật toán theo dõi ByteTrack.
├── counting/                       # Mã nguồn và logic chức năng đếm số lượng.
├── speed_estimator/                # Mã nguồn và logic chức năng ước tính tốc độ.
├── demo/                           # Các script hoặc ví dụ chạy demo.
├── evaluate/                       # Các script đánh giá hiệu suất tracking.
├── requirements.txt                # Danh sách thư viện Python cần thiết cho demo cơ bản.
├── requirements_streamlit.txt      # Danh sách thư viện Python cần thiết cho ứng dụng Streamlit.
├── LICENSE                         # Giấy phép của dự án (MIT License).
└── README.md                       # File tài liệu chính của dự án.
                  
```

## 🚀 Cài đặt và Sử dụng

Để chạy dự án, hãy làm theo các bước sau:

### 1. Tạo môi trường ảo
Việc tạo môi trường ảo sẽ giúp bạn dễ dàng quản lí các phiên bản thư viện, giúp dễ cài đặt và sửa chữa, tránh lỗi phiên bản.

``` bash
python -m venv venv
```

Tạo môi trường ảo với tên venv. Sau khi khởi tạo thành công, tiến hành kích hoạt môi trường ảo:

``` bash
venv\Scripts\activate    
```
Môi trường khởi tạo thành công sẽ hiển thị tên (venv) màu xanh trước đường dẫn.

### 2. Clone Repository

```bash
git clone https://github.com/HuynhNghiaKHMT/Vehicle_Tracking_Counting_Speeding.git
cd Vehicle_Tracking_Counting_Speeding
```

### 3. Cài đặt các thư viện cần thiết
Đối với Demo ByteTrack cơ bản:
```bash
pip install -r requirements.txt
```

Đối với Demo với ứng dụng Streamlit:
```bash
pip install -r requirements_streamlit.txt
```

## 🏃 Demo
### 1. Chạy Demo ByteTrack cơ bản
Sau khi cài đặt các thư viện trong requirements.txt:
```bash
python demo_ByteTrack.py
```
Lệnh này sẽ chạy demo tracking, counting và speeding estimation trực tiếp trên video hoặc luồng dữ liệu cấu hình sẵn.

### 2. Chạy Demo với ứng dụng Streamlit
Sau khi cài đặt các thư viện trong requirements_streamlit.txt:
```bash
python -m streamlit run streamlit_app_xlsx.py
```
Lệnh này sẽ chạy demo tracking, counting và speeding estimation trực tiếp trên web và hỗ trợ xuất thống kê `vehicle_information.xlsx`.

## 🎞️ Video Demo
Dưới đây là một đoạn video/GIF ngắn minh họa hoạt động của ứng dụng Tracking, Counting và Speeding Estimation:

<img src="assets/demoHD.gif" width="100%">


## 📊 Đánh giá hiệu suất
Để đánh giá hiệu suất của hệ thống tracking, bạn có thể sử dụng các script sau:

### 1. `convert.py`:
* Chức năng: Chuyển đổi định dạng các file annotation từ công cụ gắn nhãn sang định dạng tương thích với đầu ra của mô hình (thường là định dạng MOT Challenge hoặc tương tự).
* Đầu ra: Các file `anno_gt_videox.txt` (trong đó x là trạng thái hoặc ID của video), chứa ground truth annotations.

### 2. `run_<tên kỹ thuật tracking>_<tên mô hình detect).py`:
* Chức năng: Sử dụng mô hình tracking để xử lý video và tạo ra các dự đoán theo dõi.
* Đầu ra: Các file trong thư mục `annotations/predicted`, chứa các kết quả tracking dự đoán của mô hình.

### 3.`evaluate_<tên kỹ thuật tracking>_<tên mô hình detect).py`:
* Chức năng: Thực hiện đánh giá hiệu suất dựa trên các metrics chuẩn.
* Metrics:
    * MOTA (Multiple Object Tracking Accuracy): Một metric tổng thể, xem xét số lượng false positives, false negatives, và ID switches.
    * MOTP (Multiple Object Tracking Precision): Đo lường độ chính xác của vị trí bounding box.
    * IDF1 (ID F1 Score): Đánh giá hiệu suất duy trì ID qua thời gian.

## 📬 Thông tin thành viên nhóm

| Họ và Tên         | MSSV         | Email                        | GitHub                                      |
|-------------------|--------------|------------------------------|---------------------------------------------|
| Phạm Hồ Trúc Linh | 22520777      | 22520777@gm.uit.eduvn        | [PHTLing](https://github.com/PHTLing)                       |
| Huỳnh Trung Nghĩa | 22520945     | 22520945@gm.uit.edu.vn       | [HuynhNghiaKHMT](https://github.com/HuynhNghiaKHMT) |


## 💖 Lời cảm ơn

Chúng mình xin gửi lời cảm ơn chân thành đến cộng đồng mã nguồn mở và các tác giả đã phát triển những thư viện tuyệt vời như YOLO, SORT, DeepSort và ByteTrack. Nhờ những công cụ đó mà bọn mình có thể học hỏi, thử nghiệm và hoàn thành đồ án này. Cảm ơn các thầy cô và bạn bè đã hỗ trợ, góp ý trong suốt quá trình thực hiện.


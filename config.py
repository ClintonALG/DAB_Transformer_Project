import platform
import torch

class Config:
    # 📂 Đường dẫn dữ liệu local (Ân kiểm tra lại các folder train/val/test trên máy mình nhé)
    TRAIN_PATH = r"C:\Users\clint\source\repos\DeepLearning\Data_AmThanh\Dataset_Splitted\train"
    VAL_PATH   = r"C:\Users\clint\source\repos\DeepLearning\Data_AmThanh\Dataset_Splitted\val"
    TEST_PATH  = r"C:\Users\clint\source\repos\DeepLearning\Data_AmThanh\Dataset_Splitted\test"
    
    # 💾 Thư mục lưu checkpoint (.pt)
    SAVE_DIR = "./checkpoints"
    MAX_SAMPLES = 160000  # Giới hạn tối đa 10 giây để bảo vệ an toàn cho 4GB VRAM
    
    # 🧠 Cấu hình kiến trúc siêu nhẹ theo lời khuyên tối ưu hóa thực tế cho GTX 1650
    D_MODEL = 192
    NHEAD = 4
    NUM_LAYERS = 2
    
    # 🚀 Cấu hình huấn luyện tối ưu tốc độ phản hồi phần cứng
    BATCH_SIZE = 2           
    ACCUMULATION_STEPS = 4   # Gom 4 batch (tương đương Batch 8 ảo) mới cập nhật trọng số
    LR = 5e-5
    NUM_EPOCHS = 30
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Đánh giá: WER/CER trên val (train), không trên train — giới hạn batch để nhanh mỗi epoch
    VAL_EVAL_MAX_BATCHES = 40
    NUM_WORKERS = 0 if platform.system() == "Windows" else 4
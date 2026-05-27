"""

GIAI ĐOẠN 0 — Cấu hình toàn cục

"""

import platform

import torch



class Config:

    # --- Dữ liệu GỐC (preprocess_resample.py) ---

    RAW_TRAIN_PATH = r"C:\Users\clint\source\repos\DeepLearning\Data_AmThanh\Dataset_Splitted\train"

    RAW_VAL_PATH   = r"C:\Users\clint\source\repos\DeepLearning\Data_AmThanh\Dataset_Splitted\val"

    RAW_TEST_PATH  = r"C:\Users\clint\source\repos\DeepLearning\Data_AmThanh\Dataset_Splitted\test"



    # --- Dữ liệu 16 kHz (train / val / test) ---

    TRAIN_PATH = r"C:\Users\clint\source\repos\DeepLearning\Data_AmThanh\Dataset_Splitted_16k\train"

    VAL_PATH   = r"C:\Users\clint\source\repos\DeepLearning\Data_AmThanh\Dataset_Splitted_16k\val"

    TEST_PATH  = r"C:\Users\clint\source\repos\DeepLearning\Data_AmThanh\Dataset_Splitted_16k\test"



    TARGET_SAMPLE_RATE = 16000

    SAVE_DIR = "./checkpoints"



    # Giới hạn độ dài audio (samples @ 16kHz)

    MAX_SAMPLES = 160000       # ~10s — val / test

    MAX_SAMPLES_TRAIN = 120000  # ~7.5s — train (giảm spike VRAM khi bucket toàn file dài)



    # --- Kiến trúc ---

    D_MODEL = 192

    NHEAD = 4

    NUM_LAYERS = 2



    # --- Huấn luyện ---

    # OOM trên 4GB: đổi BATCH_SIZE=4, ACCUMULATION_STEPS=4 (giữ effective batch=16)

    BATCH_SIZE = 8

    ACCUMULATION_STEPS = 2

    LR = 3e-4

    NUM_EPOCHS = 30

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    USE_GRADIENT_CHECKPOINT = False



    # --- SpecAugment (augment.py, chỉ khi train) ---

    USE_SPEC_AUGMENT = True

    SPEC_FREQ_MASK_MAX = 24   # ~12% D_MODEL=192

    SPEC_TIME_MASK_MAX = 40

    SPEC_NUM_FREQ_MASKS = 2

    SPEC_NUM_TIME_MASKS = 2



    # --- Decode ---
    # - VAL_DECODER: dùng trong train.py khi evaluate mỗi epoch (ưu tiên nhanh)
    # - TEST_DECODER: dùng trong test_inference.py để lấy điểm báo cáo
    VAL_DECODER = "greedy"
    TEST_DECODER = "beam"
    BEAM_SIZE = 5



    # --- DataLoader ---

    VAL_EVAL_MAX_BATCHES = 50

    NUM_WORKERS = 4 if platform.system() == "Windows" else 6

    PREFETCH_FACTOR = 2

    BUCKET_BATCHING = True

    # Không gom >N mẫu dài vào cùng batch (độ dài wav samples)

    BUCKET_MAX_WAV_LEN = MAX_SAMPLES_TRAIN



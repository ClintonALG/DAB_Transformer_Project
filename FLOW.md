# Sơ đồ luồng — DAB Transformer (ASR + CTC)

## 1. Tổng quan dự án

```mermaid
flowchart TB
    subgraph Data["Dữ liệu L2-Arctic"]
        TRAIN["train/"]
        VAL["val/"]
        TEST["test/"]
    end

    subgraph Scripts["Scripts"]
        T["train.py"]
        TI["test_inference.py"]
        P["plot_metrics.py"]
    end

    TRAIN --> T
    VAL --> T
    TEST --> TI
    T --> CKPT[("checkpoints/model_e*.pt")]
    CKPT --> TI
    CKPT --> P
    P --> PNG["full_report_metrics.png"]
```

## 2. Luồng huấn luyện (`train.py`)

```mermaid
flowchart TD
    A[Bắt đầu] --> B{Checkpoint cũ?}
    B -->|Có| C[Nạp model + optimizer + scheduler + scaler]
    B -->|Không| D[Khởi tạo mới]
    C --> E[DataLoader: TRAIN + VAL]
    D --> E

    E --> F[Vòng Epoch]
    F --> G[Train mode — batch từ TRAIN]
    G --> H[WAV → Model → Logits]
    H --> I[CTC Loss / accumulation]
    I --> J{NaN/Inf?}
    J -->|Có| G
    J -->|Không| K[Backward + AMP scaler]
    K --> L{Đủ ACCUMULATION_STEPS?}
    L -->|Có| M[Clip grad + optimizer.step]
    L -->|Không| G
    G --> N{Còn gradient tồn?}
    N -->|Có| M
    M --> G

    G --> O[Eval trên VAL — WER/CER]
    O --> P[ReduceLROnPlateau theo Val WER]
    P --> Q[Lưu checkpoint: loss, wer, cer]
    Q --> F
    F --> R[Kết thúc]
```

## 3. Kiến trúc mô hình (`model.py`)

```mermaid
flowchart LR
    WAV["Waveform\n[B, T_audio]"] --> ENC["CNN Encoder\nstride 8 × 4 = ÷32"]
    ENC --> SEQ["Sequence\n[B, T', D_MODEL]"]
    SEQ --> PE["Positional Encoding"]
    PE --> B1["DAB_Block × N"]

    subgraph DAB_Block["Một DAB_Block"]
        direction TB
        ATTN["Multi-Head Self-Attention"]
        GATE["Gating Conv\n(lọc disfluency)"]
        FFN["Feed-Forward + GELU"]
        ATTN --> GATE --> FFN
    end

    B1 --> CLS["Linear Classifier"]
    CLS --> LOG["Logits [B, T', num_classes]"]
    LOG --> CTC["CTC Loss (train)"]
    LOG --> DEC["Greedy Decoder (eval)"]
```

## 4. Luồng test (`test_inference.py`)

```mermaid
flowchart TD
    A["python test_inference.py 5"] --> B{Checkpoint tồn tại?}
    B -->|Không| X[Lỗi — thoát]
    B -->|Có| C[Nạp model_e5.pt]
    C --> D[DataLoader TEST — shuffle=False]
    D --> E[Forward toàn bộ test set]
    E --> F[Greedy CTC decode]
    F --> G[Tính WER / CER trung bình]
    G --> H[In 10 mẫu minh họa]
```

## 5. Giải mã CTC (`utils.py`)

```mermaid
flowchart LR
    L["Logits per frame"] --> AM["argmax từng frame"]
    AM --> BL["Bỏ blank = 0"]
    BL --> DD["Gộp ký tự lặp liên tiếp"]
    DD --> TXT["Chuỗi a-z + space"]
```

## 6. File và vai trò

| File | Vai trò |
|------|---------|
| `config.py` | Đường dẫn train/val/test, hyperparameters |
| `dataset.py` | Đọc WAV + transcript, resample 16 kHz |
| `model.py` | DAB_Transformer |
| `utils.py` | TextProcess, CTC lengths, decode, `evaluate_metrics` |
| `train.py` | Huấn luyện CTC; metric trên **val** |
| `test_inference.py` | Metric trên **test** + mẫu minh họa |
| `plot_metrics.py` | Vẽ loss + WER/CER từ checkpoint |

## 7. Lệnh chạy

```bash
python train.py
python test_inference.py 10
python plot_metrics.py
```

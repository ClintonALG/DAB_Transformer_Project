"""
plot_metrics.py — Giai đoạn báo cáo sau train

Đọc loss / Val WER / Val CER từ các checkpoint → vẽ biểu đồ.
Không load dữ liệu âm thanh, không chạy mô hình.
"""
import torch
import matplotlib.pyplot as plt
import glob, os, re
from config import Config

def draw_full_report():
    """Đọc checkpoints/model_e*.pt và lưu full_report_metrics.png."""
    checkpoints = glob.glob(os.path.join(Config.SAVE_DIR, "model_e*.pt"))
    checkpoints.sort(key=lambda f: int(re.search(r'e(\d+)', f).group(1)))

    epochs, losses, wers, cers = [], [], [], []

    for cp in checkpoints:
        data = torch.load(cp, map_location='cpu')
        if isinstance(data, dict) and 'loss' in data:
            epochs.append(data['epoch'])
            losses.append(data['loss'])
            wers.append(data.get('wer', 1.0))
            cers.append(data.get('cer', 1.0))

    if not epochs:
        print("❌ Chưa có dữ liệu chỉ số trong checkpoint!")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    ax1.plot(epochs, losses, 'r-o', label='Training Loss')
    ax1.set_title('Sơ đồ độ hội tụ mô hình (Loss Curve)')
    ax1.set_ylabel('Loss Value')
    ax1.grid(True)
    ax1.legend()

    ax2.plot(epochs, [w*100 for w in wers], 'b-s', label='Val WER (Lỗi từ)')
    ax2.plot(epochs, [c*100 for c in cers], 'g-d', label='Val CER (Lỗi ký tự)')
    ax2.set_title('Sơ đồ tỷ lệ lỗi trên tập Val (Error Rate)')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Percentage (%)')
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plt.savefig('full_report_metrics.png')
    print("✅ Đã lưu bộ sơ đồ tại: full_report_metrics.png")
    plt.show()

if __name__ == "__main__":
    draw_full_report()

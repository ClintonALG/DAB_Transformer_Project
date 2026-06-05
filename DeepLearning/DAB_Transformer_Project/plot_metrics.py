"""
plot_metrics.py — Trực quan hóa toàn diện lịch sử huấn luyện
"""
import os
import glob
import re
import torch
import matplotlib.pyplot as plt

from config import Config

def plot_all_metrics():
    # 1. Quét tìm tất cả các file checkpoint
    checkpoints = glob.glob(os.path.join(Config.SAVE_DIR, "model_e*.pt"))
    if not checkpoints:
        print("❌ Không tìm thấy checkpoint nào để vẽ.")
        return

    print(f"🔍 Đang tổng hợp dữ liệu từ {len(checkpoints)} epochs...")

    epochs = []
    losses = []
    wers = []
    cers = []

    # 2. Đọc dữ liệu từ từng file
    for ckpt_path in checkpoints:
        try:
            # Chỉ cần load meta data, không cần load toàn bộ weights model (nhanh hơn)
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
            if isinstance(ckpt, dict):
                # Trích xuất epoch từ tên file để đảm bảo thứ tự chính xác
                match = re.search(r"model_e(\d+)\.pt", ckpt_path)
                if match:
                    epoch = int(match.group(1))
                else:
                    epoch = ckpt.get("epoch", 0)

                epochs.append(epoch)
                # Lấy loss, nếu không có thì mặc định là 0
                losses.append(ckpt.get("loss", 0.0))
                # Lấy WER và CER, nhân 100 để hiển thị dạng phần trăm
                wers.append(ckpt.get("wer", 0.0) * 100)
                cers.append(ckpt.get("cer", 0.0) * 100)
        except Exception as e:
            print(f"⚠️ Bỏ qua file lỗi {ckpt_path}: {e}")

    # Sắp xếp lại theo thứ tự epoch
    sorted_indices = sorted(range(len(epochs)), key=lambda k: epochs[k])
    epochs = [epochs[i] for i in sorted_indices]
    losses = [losses[i] for i in sorted_indices]
    wers = [wers[i] for i in sorted_indices]
    cers = [cers[i] for i in sorted_indices]

    # 3. Bắt đầu vẽ biểu đồ
    fig, axes = plt.subplots(2, 1, figsize=(15, 12)) # 2 hàng, 1 cột

    # Biểu đồ 1: Training Loss
    axes[0].plot(epochs, losses, marker='o', color='red', label='Training Loss')
    axes[0].set_title('Sơ đồ độ hội tụ mô hình (Loss Curve)', fontsize=14)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss Value', fontsize=12)
    axes[0].grid(True)
    axes[0].legend(fontsize=12)

    # Biểu đồ 2: Validation WER & CER
    axes[1].plot(epochs, wers, marker='s', linestyle='-', color='blue', label='Val WER (Phoneme Error Rate)')
    axes[1].plot(epochs, cers, marker='d', linestyle='--', color='green', label='Val CER (Character Error Rate)')
    axes[1].set_title('Sơ đồ tỷ lệ lỗi trên tập Validation', fontsize=14)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Error Rate (%)', fontsize=12)
    axes[1].grid(True)
    axes[1].legend(fontsize=12)

    # Nếu mô hình có Early Stopping, làm nổi bật điểm tốt nhất (Best WER)
    best_wer = min(wers)
    best_epoch_index = wers.index(best_wer)
    best_epoch = epochs[best_epoch_index]
    
    # Khoanh tròn điểm tốt nhất trên đồ thị
    axes[1].scatter(best_epoch, best_wer, color='gold', s=150, zorder=5, edgecolors='black', label=f'Best Model (Epoch {best_epoch})')
    axes[1].legend(fontsize=12)

    plt.tight_layout()
    
    # Lưu file
    out_file = 'full_report_metrics.png'
    plt.savefig(out_file, dpi=150)
    print(f"✅ Đã lưu biểu đồ tổng hợp tại: {out_file}")

if __name__ == '__main__':
    plot_all_metrics()
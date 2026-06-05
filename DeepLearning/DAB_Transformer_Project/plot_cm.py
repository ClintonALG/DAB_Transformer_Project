import os
import glob
import re
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from config import Config
from model import DAB_Transformer
from dataset import L2ArcticDataset, make_dataloader
from utils import text_process, decode_logits, ARPABET_PHONEMES

def get_best_checkpoint():
    """Tự động tìm file checkpoint có Val WER thấp nhất."""
    checkpoints = glob.glob(os.path.join(Config.SAVE_DIR, "model_e*.pt"))
    if not checkpoints:
        return None, 0, float('inf')
    
    best_ckpt = None
    best_wer = float('inf')
    best_epoch = 0

    print("🔍 Đang quét các checkpoint để tìm model tốt nhất...")
    for ckpt_path in checkpoints:
        try:
            # Load cực nhanh chỉ để đọc metadata
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
            if isinstance(ckpt, dict) and "wer" in ckpt:
                if ckpt["wer"] < best_wer:
                    best_wer = ckpt["wer"]
                    best_ckpt = ckpt_path
                    best_epoch = ckpt["epoch"]
        except Exception as e:
            print(f"⚠️ Lỗi khi đọc file {ckpt_path}: {e}")

    # Fallback: Nếu các file không lưu "wer", lấy epoch cao nhất
    if best_ckpt is None:
        print("⚠️ Không tìm thấy metadata 'wer' trong checkpoint, sẽ lấy epoch cao nhất.")
        best_ckpt = max(checkpoints, key=lambda x: int(re.search(r"model_e(\d+)\.pt", x).group(1)))
        best_epoch = int(re.search(r"model_e(\d+)\.pt", best_ckpt).group(1))

    return best_ckpt, best_epoch, best_wer

def generate_confusion_matrix():
    # 1. Tìm checkpoint TỐT NHẤT
    ckpt_path, epoch_num, best_wer = get_best_checkpoint()
    if not ckpt_path:
        print("❌ Không tìm thấy checkpoint nào trong thư mục!")
        return

    print(f"🚀 Đang nạp BEST MODEL từ: {ckpt_path} (Epoch: {epoch_num} | WER: {best_wer*100:.2f}%)")
    
    model = DAB_Transformer(len(text_process.char_map), Config.D_MODEL, Config.NHEAD, Config.NUM_LAYERS).to(Config.DEVICE)
    ckpt = torch.load(ckpt_path, map_location=Config.DEVICE, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt)
    model.eval()

    test_loader = make_dataloader(L2ArcticDataset(Config.TEST_PATH, text_process), shuffle=False)
    
    all_targets = []
    all_preds = []
    
    print("📊 Đang infer tập Test để thu thập dữ liệu (có thể mất vài phút)...")
    with torch.no_grad():
        for wavs, labels, w_lens, l_lens in test_loader:
            logits, _ = model(wavs.to(Config.DEVICE))
            preds = decode_logits(logits, w_lens, text_process)
            for idx in range(len(preds)):
                target = text_process.labels_to_text(labels[idx], l_lens[idx].item())
                pred = preds[idx]
                
                t_list = [p for p in target.split() if p in ARPABET_PHONEMES]
                p_list = [p for p in pred.split() if p in ARPABET_PHONEMES]
                min_len = min(len(t_list), len(p_list))
                if min_len > 0:
                    all_targets.extend(t_list[:min_len])
                    all_preds.extend(p_list[:min_len])

    if len(all_targets) == 0:
        print("❌ Lỗi: Không thu thập được dữ liệu nào từ tập Test!")
        return

    # Lọc danh sách âm vị để vẽ (Bỏ blank và space)
    labels_clean = [p for p in ARPABET_PHONEMES if p not in ["<blank>", " "]]
    
    # Tính ma trận (số lượng đếm)
    cm = confusion_matrix(all_targets, all_preds, labels=labels_clean)
    
    # Chuyển đổi thành dạng chuẩn hóa (Normalized) giống ảnh bạn gửi
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # Xử lý các dòng bị NaN (nếu có nhãn không xuất hiện trong test set)
    cm_normalized = np.nan_to_num(cm_normalized)

    # 3. Vẽ ma trận
    plt.figure(figsize=(24, 20)) # Tăng kích thước để dễ đọc số
    sns.heatmap(
        cm_normalized, 
        annot=True,              # Hiển thị số
        fmt=".2f",               # Định dạng số thập phân (vd: 0.50)
        cmap='Blues',            # Màu xanh như ảnh của bạn
        xticklabels=labels_clean, 
        yticklabels=labels_clean,
        cbar_kws={'label': 'Tỷ lệ dự đoán'}
    )
    
    plt.title(f'Normalized Confusion Matrix - Best Epoch {epoch_num} (WER: {best_wer*100:.2f}%)', fontsize=16)
    plt.xlabel('Dự đoán (Máy)', fontsize=14)
    plt.ylabel('Chuẩn (Target)', fontsize=14)
    plt.xticks(rotation=45)
    
    out_file = f'confusion_matrix_best_e{epoch_num}.png'
    plt.savefig(out_file, bbox_inches='tight', dpi=150)
    print(f"✅ Đã lưu ma trận nhầm lẫn chuẩn hóa tại: {out_file}")

if __name__ == '__main__':
    generate_confusion_matrix()
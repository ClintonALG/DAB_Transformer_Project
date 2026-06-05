"""
test_inference.py — Cỗ máy bắt lỗi phát âm (Chấm điểm trực tiếp File Giọng thật)
"""
import os
import argparse
import json
import torch
import torchaudio
import librosa
import jiwer
import numpy as np

from config import Config
from model import DAB_Transformer
from utils import text_process, decode_logits

def build_pronunciation_feedback(target, predicted):
    """
    Sử dụng Jiwer Alignment để gióng hàng và phân tích lỗi đọc.
    Trả về cấu trúc JSON sẵn sàng cho Frontend.
    """
    target_clean = " ".join(target.split())
    pred_clean = " ".join(predicted.split())
    
    if not pred_clean:
        return [{"status": "error", "message": "Không nhận diện được giọng nói"}]
        
    alignment = jiwer.process_words(target_clean, pred_clean).alignments[0]
    t_words = target_clean.split()
    p_words = pred_clean.split()
    feedback_list = []
    
    for chunk in alignment:
        error_type = chunk.type
        if error_type == 'equal':
            for i in range(chunk.ref_start_idx, chunk.ref_end_idx):
                feedback_list.append({
                    "phoneme": t_words[i], 
                    "status": "correct", 
                    "predicted": t_words[i]
                })
        elif error_type == 'substitute':
            for i, j in zip(range(chunk.ref_start_idx, chunk.ref_end_idx), range(chunk.hyp_start_idx, chunk.hyp_end_idx)):
                feedback_list.append({
                    "phoneme": t_words[i], 
                    "status": "substitution", 
                    "predicted": p_words[j],
                    "message": f"Nhầm /{t_words[i]}/ thành /{p_words[j]}/"
                })
        elif error_type == 'delete':
            for i in range(chunk.ref_start_idx, chunk.ref_end_idx):
                feedback_list.append({
                    "phoneme": t_words[i], 
                    "status": "deletion", 
                    "predicted": "-",
                    "message": f"Bạn bị nuốt âm /{t_words[i]}/"
                })
        elif error_type == 'insert':
            for j in range(chunk.hyp_start_idx, chunk.hyp_end_idx):
                feedback_list.append({
                    "phoneme": "-", 
                    "status": "insertion", 
                    "predicted": p_words[j],
                    "message": f"Phát âm thừa âm /{p_words[j]}/"
                })

    return feedback_list

def load_and_resample_audio(file_path, target_sr=16000):
    """
    Tự động đọc file wav và ép về 16kHz
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Không tìm thấy file audio: {file_path}")
        
    # Dùng librosa load để dễ dàng resample tự động từ mọi SR
    wav_numpy, sr = librosa.load(file_path, sr=target_sr)
    
    # Chuyển numpy array thành Torch Tensor shape [1, T]
    wav_tensor = torch.FloatTensor(wav_numpy).unsqueeze(0)
    
    return wav_tensor

def test_real_audio(epoch_num, audio_path, target_text):
    """
    Chạy file ghi âm thực tế qua mô hình
    """
    checkpoint_path = f"{Config.SAVE_DIR}/model_e{epoch_num}.pt"
    if not os.path.exists(checkpoint_path):
        print(f"❌ Không tìm thấy file checkpoint: {checkpoint_path}")
        return

    # 1. Nạp Model
    model = DAB_Transformer(
        len(text_process.char_map), Config.D_MODEL, Config.NHEAD, Config.NUM_LAYERS
    ).to(Config.DEVICE)

    checkpoint = torch.load(checkpoint_path, map_location=Config.DEVICE, weights_only=True)
    state = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state)
    model.eval()

    # 2. Xử lý dữ liệu đầu vào
    print(f"\n🎤 Đang xử lý file: {audio_path}")
    print(f"🎯 Đoạn văn chuẩn (Target): '{target_text}'\n")
    
    try:
        wav = load_and_resample_audio(audio_path, Config.TARGET_SAMPLE_RATE)
        wav = wav.to(Config.DEVICE)
        
        # Lấy độ dài audio
        w_lens = torch.tensor([wav.shape[1]], dtype=torch.long)
        
        # Chuyển text mục tiêu sang chuỗi âm vị (phonemes)
        target_phonemes = " ".join(text_process.text_to_phonemes(target_text))

    except Exception as e:
        print(f"Lỗi khi xử lý file audio: {e}")
        return

    # 3. Chạy Inference
    with torch.no_grad():
        logits, _ = model(wav)
        pred_phonemes = decode_logits(logits, w_lens, text_process)[0]
    
    # 4. Trích xuất phản hồi
    feedback_json = build_pronunciation_feedback(target_phonemes, pred_phonemes)
    
    print("================ KẾT QUẢ ĐÁNH GIÁ ================")
    if pred_phonemes.strip():
        print(jiwer.visualize_alignment(jiwer.process_words(target_phonemes, pred_phonemes)))
    
    print("\n[📦 Dữ liệu JSON Backend trả về App Flutter]:")
    print(json.dumps(feedback_json, indent=2, ensure_ascii=False))
    print("\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=50, help="Số epoch checkpoint (mặc định 50)")
    parser.add_argument("--audio", type=str, required=True, help="Đường dẫn đến file .wav của bạn")
    parser.add_argument("--text", type=str, required=True, help="Câu tiếng Anh mà bạn đang đọc trong file audio")
    
    args = parser.parse_args()
    test_real_audio(epoch_num=args.epoch, audio_path=args.audio, target_text=args.text)
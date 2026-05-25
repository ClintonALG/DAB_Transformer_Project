import os
import sys
import torch
from torch.utils.data import DataLoader
from config import Config
from model import DAB_Transformer
from dataset import L2ArcticDataset, collate_fn
from utils import text_process, greedy_decoder, evaluate_metrics

def test_model(epoch_num=None, show_samples=10):
    if epoch_num is None:
        if len(sys.argv) > 1:
            epoch_num = sys.argv[1].strip()
        else:
            epoch_num = input("➡️ Nhập số Epoch bạn muốn test (ví dụ: 1, 5, 10): ").strip()

    checkpoint_path = f"{Config.SAVE_DIR}/model_e{epoch_num}.pt"
    if not os.path.exists(checkpoint_path):
        print(f"❌ Không tìm thấy file checkpoint: {checkpoint_path}")
        return

    model = DAB_Transformer(
        len(text_process.char_map), Config.D_MODEL, Config.NHEAD, Config.NUM_LAYERS
    ).to(Config.DEVICE)

    checkpoint = torch.load(checkpoint_path, map_location=Config.DEVICE)
    state = (
        checkpoint["model_state_dict"]
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
        else checkpoint
    )
    model.load_state_dict(state)
    model.eval()

    loader_kw = {
        "batch_size": 1,
        "collate_fn": collate_fn,
        "num_workers": Config.NUM_WORKERS,
        "pin_memory": Config.DEVICE.type == "cuda",
    }
    test_loader = DataLoader(
        L2ArcticDataset(Config.TEST_PATH, text_process), shuffle=False, **loader_kw
    )

    print(f"\n🎧 Đánh giá toàn bộ tập TEST — Epoch {epoch_num}...")
    avg_wer, avg_cer, n_samples = evaluate_metrics(model, test_loader, Config.DEVICE)
    print(f"\n📊 Kết quả TEST ({n_samples} mẫu):")
    print(f"   WER: {avg_wer * 100:.2f}%")
    print(f"   CER: {avg_cer * 100:.2f}%")

    if show_samples > 0:
        print(f"\n--- {show_samples} mẫu minh họa ---")
        with torch.no_grad():
            for i, (wavs, labels, _, l_lens) in enumerate(test_loader):
                wavs = wavs.to(Config.DEVICE)
                logits, _ = model(wavs)
                pred = greedy_decoder(logits[0], text_process)
                target = text_process.int_to_text(labels[0][: l_lens[0]].tolist())
                print(f"\n[Mẫu {i + 1}]")
                print(f"Gốc: {target}")
                print(f"Máy: {pred if pred else '(Không nhận diện được)'}")
                if i + 1 >= show_samples:
                    break

if __name__ == "__main__":
    test_model()

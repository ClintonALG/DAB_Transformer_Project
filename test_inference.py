"""
test_inference.py — Suy luận & đánh giá trên tập TEST

Giai đoạn:
  [I1] Nạp checkpoint đã train
  [I2] DataLoader test (cùng tiền xử lý dataset.py [B][C])
  [I3] evaluate_metrics — WER/CER toàn tập test
  [I4] In mẫu minh họa

Chạy:
  python test_inference.py 10
  python test_inference.py 10 --decoder greedy
  python test_inference.py 10 --decoder beam --beam-size 8 --show-samples 5
"""

import os
import argparse

import torch
from config import Config
from model import DAB_Transformer
from dataset import L2ArcticDataset, make_dataloader
from utils import (
    text_process,
    decode_frame_logits,
    calculate_input_lengths,
    evaluate_metrics,
)


def test_model(epoch_num=None, show_samples=10, decoder=None, beam_size=None):
    # [I1] Chọn epoch & nạp trọng số
    if epoch_num is None:
        epoch_num = input("➡️ Nhập số Epoch bạn muốn test (ví dụ: 1, 5, 10): ").strip()

    decoder = (decoder or Config.TEST_DECODER).lower()
    if decoder not in {"greedy", "beam"}:
        raise ValueError("decoder phải là 'greedy' hoặc 'beam'")
    beam_size = beam_size if beam_size is not None else Config.BEAM_SIZE

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

    # [I2] Load test set
    test_loader = make_dataloader(L2ArcticDataset(Config.TEST_PATH, text_process), shuffle=False)

    # [I3] Đánh giá định lượng
    print(f"\n🎧 Đánh giá tập TEST — Epoch {epoch_num} ({decoder}, beam={beam_size})...")
    avg_wer, avg_cer, n_samples = evaluate_metrics(
        model,
        test_loader,
        Config.DEVICE,
        log_samples=3,
        decoder=decoder,
        beam_size=beam_size,
    )
    print(f"\n📊 Kết quả TEST ({n_samples} mẫu):")
    print(f"   WER: {avg_wer * 100:.2f}%")
    print(f"   CER: {avg_cer * 100:.2f}%")

    # [I4] Minh họa vài câu
    if show_samples > 0:
        print(f"\n--- {show_samples} mẫu minh họa ---")
        with torch.no_grad():
            for i, (wavs, labels, w_lens, l_lens) in enumerate(test_loader):
                wavs = wavs.to(Config.DEVICE)
                logits, _ = model(wavs)

                t = int(calculate_input_lengths(w_lens)[0].item())
                pred = decode_frame_logits(
                    logits[0, :t],
                    text_process,
                    decoder=decoder,
                    beam_size=beam_size,
                )

                target = text_process.int_to_text(labels[0][: l_lens[0]].tolist())
                print(f"\n[Mẫu {i + 1}]")
                print(f"Gốc: {target}")
                print(f"Máy: {pred if pred else '(Không nhận diện được)'}")
                if i + 1 >= show_samples:
                    break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Đánh giá mô hình trên tập TEST")
    parser.add_argument("epoch", nargs="?", help="Số epoch checkpoint, ví dụ 30")
    parser.add_argument("--decoder", choices=["greedy", "beam"], default=Config.TEST_DECODER)
    parser.add_argument("--beam-size", type=int, default=Config.BEAM_SIZE)
    parser.add_argument("--show-samples", type=int, default=10)
    args = parser.parse_args()

    test_model(
        epoch_num=args.epoch,
        show_samples=args.show_samples,
        decoder=args.decoder,
        beam_size=args.beam_size,
    )



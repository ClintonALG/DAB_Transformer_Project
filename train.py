"""

train.py — Huấn luyện ASR (CTC)



Luồng chính trong train():

  [0] Khởi tạo GPU, model, optimizer, scheduler, CTC loss

  [1] Resume checkpoint (nếu có)

  [2] Tạo DataLoader train + val  ← tiền xử lý xảy ra khi đọc batch (dataset.py)

  [3] Vòng epoch:

        [3a] Train: forward → CTC loss → backward → cập nhật trọng số

        [3b] Val:   evaluate_metrics (decode + WER/CER, không train)

        [3c] Lưu checkpoint + điều chỉnh LR theo Val WER

"""

import torch

import torch.nn.functional as F

import os

import glob

import re



from config import Config

from model import DAB_Transformer

from dataset import L2ArcticDataset, make_dataloader

from utils import text_process, calculate_input_lengths, evaluate_metrics





def _optimizer_step(scaler, optimizer, model):

    """Một bước cập nhật trọng số sau accumulation (clip grad + AdamW)."""

    scaler.unscale_(optimizer)

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    scaler.step(optimizer)

    scaler.update()

    optimizer.zero_grad()





def train():

    # --- [0] Thiết lập ---

    if not torch.cuda.is_available():

        print("⚠️ Không thấy CUDA — huấn luyện trên CPU sẽ rất chậm.")

    else:

        print(f"🖥️ GPU: {torch.cuda.get_device_name(0)}")



    torch.backends.cudnn.benchmark = True

    os.makedirs(Config.SAVE_DIR, exist_ok=True)



    model = DAB_Transformer(

        num_classes=len(text_process.char_map),

        d_model=Config.D_MODEL,

        nhead=Config.NHEAD,

        num_layers=Config.NUM_LAYERS,

    ).to(Config.DEVICE)



    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LR)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(

        optimizer, mode="min", factor=0.5, patience=2

    )

    criterion = torch.nn.CTCLoss(blank=0, zero_infinity=True)

    scaler = torch.amp.GradScaler("cuda", enabled=Config.DEVICE.type == "cuda")



    # --- [1] Resume ---

    start_epoch = 1

    checkpoints = glob.glob(os.path.join(Config.SAVE_DIR, "model_e*.pt"))

    if checkpoints:

        latest_checkpoint = max(checkpoints, key=os.path.getctime)

        print(f"🔄 Tìm thấy checkpoint cũ: {latest_checkpoint}. Đang nạp...")

        checkpoint_data = torch.load(latest_checkpoint, map_location=Config.DEVICE)



        if isinstance(checkpoint_data, dict) and "model_state_dict" in checkpoint_data:

            model.load_state_dict(checkpoint_data["model_state_dict"])

            optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])

            if "scheduler_state_dict" in checkpoint_data:

                scheduler.load_state_dict(checkpoint_data["scheduler_state_dict"])

            if "scaler_state_dict" in checkpoint_data:

                scaler.load_state_dict(checkpoint_data["scaler_state_dict"])

            start_epoch = checkpoint_data["epoch"] + 1

        else:

            model.load_state_dict(checkpoint_data)

            match = re.search(r"e(\d+)", latest_checkpoint)

            if match:

                start_epoch = int(match.group(1)) + 1

        print(f"🚀 Tiếp tục từ Epoch {start_epoch}")

    else:

        print("🆕 Không có checkpoint cũ. Huấn luyện từ đầu.")



    # --- [2] Dữ liệu: index [A] lúc init; tiền xử lý [B] mỗi batch khi train ---

    train_set = L2ArcticDataset(
        Config.TRAIN_PATH, text_process, max_samples=Config.MAX_SAMPLES_TRAIN
    )
    val_set = L2ArcticDataset(Config.VAL_PATH, text_process)

    train_loader = make_dataloader(train_set, shuffle=True)

    val_loader = make_dataloader(val_set, shuffle=False)



    n_samples = len(train_set)

    n_batches = len(train_loader)

    eff_batch = Config.BATCH_SIZE * Config.ACCUMULATION_STEPS

    print(

        f"📦 Train: {n_samples} mẫu | {n_batches} batch/epoch "

        f"(batch={Config.BATCH_SIZE}×{Config.ACCUMULATION_STEPS}={eff_batch}) | "

        f"checkpoint={'bật' if Config.USE_GRADIENT_CHECKPOINT else 'tắt'} | "

        f"bucket={'bật' if Config.BUCKET_BATCHING else 'tắt'} | "

        f"SpecAug={'bật' if Config.USE_SPEC_AUGMENT else 'tắt'} | "

        f"val_decode={Config.VAL_DECODER}(beam={Config.BEAM_SIZE})"

    )



    # --- [3] Vòng epoch ---

    for epoch in range(start_epoch, Config.NUM_EPOCHS + 1):

        # [3a] Huấn luyện

        model.train()

        epoch_loss = 0.0

        valid_batches = 0

        optimizer.zero_grad()

        accum_count = 0



        for i, (wavs, labels, w_lens, l_lens) in enumerate(train_loader):

            wavs, labels = wavs.to(Config.DEVICE), labels.to(Config.DEVICE)



            with torch.amp.autocast("cuda", enabled=Config.DEVICE.type == "cuda"):

                logits, _ = model(wavs)

                input_lengths = calculate_input_lengths(w_lens).to(Config.DEVICE)

                logits_fp32 = logits.float()

                log_probs = F.log_softmax(logits_fp32, dim=2).transpose(0, 1)

                loss = criterion(log_probs, labels, input_lengths, l_lens) / Config.ACCUMULATION_STEPS



            if torch.isnan(loss) or torch.isinf(loss):

                optimizer.zero_grad()

                accum_count = 0

                continue



            scaler.scale(loss).backward()

            accum_count += 1

            epoch_loss += loss.item() * Config.ACCUMULATION_STEPS

            valid_batches += 1



            if accum_count % Config.ACCUMULATION_STEPS == 0:

                _optimizer_step(scaler, optimizer, model)

                accum_count = 0



            if i % 200 == 0:

                print(

                    f"Epoch {epoch} | Batch {i}/{n_batches} | "

                    f"Loss: {loss.item() * Config.ACCUMULATION_STEPS:.4f}"

                )



        if accum_count > 0:

            _optimizer_step(scaler, optimizer, model)



        avg_loss = epoch_loss / valid_batches if valid_batches > 0 else float("inf")



        # [3b] Đánh giá validation (decode + WER/CER — utils.evaluate_metrics)

        print(f"📊 Val WER/CER (Epoch {epoch}) — 3 mẫu minh họa:")

        avg_wer, avg_cer, n_val = evaluate_metrics(

            model,

            val_loader,

            Config.DEVICE,

            max_batches=Config.VAL_EVAL_MAX_BATCHES,

            log_samples=3,
            decoder=Config.VAL_DECODER,
            beam_size=Config.BEAM_SIZE,

        )

        scheduler.step(avg_wer)



        # [3c] Lưu checkpoint

        torch.save(

            {

                "epoch": epoch,

                "model_state_dict": model.state_dict(),

                "optimizer_state_dict": optimizer.state_dict(),

                "scheduler_state_dict": scheduler.state_dict(),

                "scaler_state_dict": scaler.state_dict(),

                "loss": avg_loss,

                "wer": avg_wer,

                "cer": avg_cer,

                "val_samples": n_val,

            },

            f"{Config.SAVE_DIR}/model_e{epoch}.pt",

        )



        lr = optimizer.param_groups[0]["lr"]

        print(f"✅ Epoch {epoch} | Train Loss: {avg_loss:.4f} | Val WER: {avg_wer * 100:.2f}% | Val CER: {avg_cer * 100:.2f}% | LR: {lr:.2e}")

        print("-" * 60)



        if Config.DEVICE.type == "cuda":

            torch.cuda.empty_cache()





if __name__ == "__main__":

    train()



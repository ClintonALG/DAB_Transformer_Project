import torch
import torchaudio
import os
from torch.utils.data import Dataset
from config import Config

class L2ArcticDataset(Dataset):
    def __init__(self, folder_path, text_process):
        self.samples = []
        self.text_process = text_process
        self.resampler = torchaudio.transforms.Resample(orig_freq=44100, new_freq=16000)
        
        if not os.path.exists(folder_path):
            print(f"❌ Thư mục không tồn tại: {folder_path}")
            return

        speakers = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
        for spk in speakers:
            wav_dir = os.path.join(folder_path, spk, "wav")
            txt_dir = os.path.join(folder_path, spk, "transcript")
            if os.path.exists(wav_dir):
                for f in os.listdir(wav_dir):
                    if f.endswith(".wav"):
                        wav_path = os.path.join(wav_dir, f)
                        txt_path = os.path.join(txt_dir, f.replace(".wav", ".txt"))
                        if os.path.exists(txt_path):
                            self.samples.append({"audio": wav_path, "text": txt_path})
                            
        print(f"🎯 Đã nạp thành công {len(self.samples)} tệp âm thanh từ: {folder_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        wav, _ = torchaudio.load(s["audio"])
        wav = self.resampler(wav)[0]
        
        # Chiến thuật cắt ngắn âm thanh dài bảo vệ biên bộ nhớ đồ họa VRAM
        if wav.shape[0] > Config.MAX_SAMPLES:
            wav = wav[:Config.MAX_SAMPLES]
            
        with open(s["text"], 'r', encoding='utf-8') as f:
            transcript = f.read().strip()
            
        label = torch.tensor(self.text_process.text_to_int(transcript))
        return wav, label

def collate_fn(batch):
    waveforms, labels = zip(*batch)
    wav_lens = torch.tensor([w.shape[0] for w in waveforms])
    label_lens = torch.tensor([len(l) for l in labels])
    waveforms = torch.nn.utils.rnn.pad_sequence(waveforms, batch_first=True)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True)
    return waveforms, labels, wav_lens, label_lens
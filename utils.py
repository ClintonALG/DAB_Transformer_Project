import torch
from jiwer import wer, cer

class TextProcess:
    def __init__(self):
        self.char_map = {"<blank>": 0, "a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6, "g": 7, "h": 8, "i": 9, "j": 10, "k": 11, "l": 12, "m": 13, "n": 14, "o": 15, "p": 16, "q": 17, "r": 18, "s": 19, "t": 20, "u": 21, "v": 22, "w": 23, "x": 24, "y": 25, "z": 26, " ": 27}
        self.index_map = {v: k for k, v in self.char_map.items()}

    def text_to_int(self, text):
        return [self.char_map[c] for c in text.lower() if c in self.char_map]

    def int_to_text(self, indices):
        return "".join([self.index_map[i] for i in indices if i in self.index_map and i != 0])

text_process = TextProcess()

def calculate_input_lengths(w_lens):
    # Công thức toán học tính toán chính xác chuỗi sequence length sau nén Stride 32 (8 x 4)
    l = ((w_lens + 2*5 - 11) // 8) + 1
    l = ((l + 2*3 - 7) // 4) + 1
    return l

def greedy_decoder(output, text_process):
    arg_maxes = torch.argmax(output, dim=-1)
    decodes = []
    for i in range(len(arg_maxes)):
        if arg_maxes[i] != 0:
            if i == 0 or arg_maxes[i] != arg_maxes[i-1]:
                decodes.append(arg_maxes[i].item())
    return text_process.int_to_text(decodes)

def evaluate_metrics(model, data_loader, device, max_batches=None):
    """Tính WER/CER trung bình trên một DataLoader (thường là val hoặc test)."""
    model.eval()
    total_wer, total_cer, samples = 0.0, 0.0, 0
    with torch.no_grad():
        for batch_idx, (wavs, labels, _, l_lens) in enumerate(data_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            logits, _ = model(wavs.to(device))
            for idx in range(len(wavs)):
                pred = greedy_decoder(logits[idx], text_process)
                target = text_process.int_to_text(labels[idx][: l_lens[idx]].tolist())
                if target:
                    total_wer += wer(target, pred)
                    total_cer += cer(target, pred)
                    samples += 1
    if samples == 0:
        return 1.0, 1.0, 0
    return total_wer / samples, total_cer / samples, samples
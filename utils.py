"""

utils.py — Chuẩn hóa text, CTC, decode (greedy / beam), đánh giá



  [T1] TextProcess.normalize / text_to_int

  [T2] calculate_input_lengths

  [T3] greedy_decoder, ctc_beam_search, decode_logits

  [T4] evaluate_metrics

"""

import re

import math

import torch

import torch.nn.functional as F

from jiwer import wer, cer

from config import Config



# Số → chữ (0–999) không cần thư viện ngoài

_ONES = (

    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",

    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",

    "seventeen", "eighteen", "nineteen",

)

_TENS = ("", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety")





def _int_to_english(n: int) -> str:

    if n < 0:

        return str(n)

    if n < 20:

        return _ONES[n]

    if n < 100:

        t, o = divmod(n, 10)

        return _TENS[t] if o == 0 else f"{_TENS[t]} {_ONES[o]}"

    if n < 1000:

        h, r = divmod(n, 100)

        head = f"{_ONES[h]} hundred"

        return head if r == 0 else f"{head} {_int_to_english(r)}"

    return str(n)





class TextProcess:

    """

    [T1] Chuẩn hóa transcript trước khi gán nhãn CTC:

      - Số → chữ (2 → two)

      - Dấu câu / ký tự đặc biệt → khoảng trắng

      - Lowercase, chỉ giữ a-z và space

    """



    def __init__(self):

        self.char_map = {

            "<blank>": 0, "a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6, "g": 7,

            "h": 8, "i": 9, "j": 10, "k": 11, "l": 12, "m": 13, "n": 14, "o": 15,

            "p": 16, "q": 17, "r": 18, "s": 19, "t": 20, "u": 21, "v": 22, "w": 23,

            "x": 24, "y": 25, "z": 26, " ": 27,

        }

        self.index_map = {v: k for k, v in self.char_map.items()}



    def normalize(self, text: str) -> str:

        """Chuẩn hóa chuỗi thô từ file transcript."""

        text = text.lower().strip()



        def _num_sub(match):

            return _int_to_english(int(match.group()))



        text = re.sub(r"\b\d+\b", _num_sub, text)

        text = re.sub(r"[^a-z\s]", " ", text)

        text = re.sub(r"\s+", " ", text).strip()

        return text



    def text_to_int(self, text):

        text = self.normalize(text)

        return [self.char_map[c] for c in text if c in self.char_map]



    def int_to_text(self, indices):

        return "".join([self.index_map[i] for i in indices if i in self.index_map and i != 0])



    def labels_to_text(self, label_tensor, length):

        """Chuyển tensor label đã pad → chuỗi (để so WER)."""

        return self.int_to_text(label_tensor[:length].tolist())





text_process = TextProcess()





def calculate_input_lengths(w_lens):

    """[T2] Số frame logits sau CNN (stride 8×4)."""

    l = ((w_lens + 2 * 5 - 11) // 8) + 1

    l = ((l + 2 * 3 - 7) // 4) + 1

    return l





def _log_add(a, b):

    if a == float("-inf"):

        return b

    if b == float("-inf"):

        return a

    if a > b:

        return a + math.log1p(math.exp(b - a))

    return b + math.log1p(math.exp(a - b))





def ctc_beam_search(log_probs, beam_width=5, blank=0):

    """

    [T3] CTC prefix beam search (Graves et al.).

    log_probs: [T, V] log-softmax trên CPU.

    Trả về list token id (không gồm blank).

    """

    t_steps, _ = log_probs.shape

    beams = {(): (0.0, float("-inf"))}



    for t in range(t_steps):

        lp = log_probs[t]

        next_beams = {}

        ranked = sorted(

            beams.items(),

            key=lambda kv: _log_add(kv[1][0], kv[1][1]),

            reverse=True,

        )[:beam_width]



        for prefix, (pb, pnb) in ranked:

            # Kết thúc bằng blank

            nb = next_beams.get(prefix, (float("-inf"), float("-inf")))

            next_beams[prefix] = (

                _log_add(nb[0], lp[blank].item() + _log_add(pb, pnb)),

                nb[1],

            )



            for c in range(lp.shape[0]):

                if c == blank:

                    continue

                logp = lp[c].item()

                if prefix and prefix[-1] == c:

                    nb = next_beams.get(prefix, (float("-inf"), float("-inf")))

                    next_beams[prefix] = (

                        nb[0],

                        _log_add(nb[1], logp + pb + pnb),

                    )

                else:

                    new_p = prefix + (c,)

                    nb = next_beams.get(new_p, (float("-inf"), float("-inf")))

                    next_beams[new_p] = (

                        nb[0],

                        _log_add(nb[1], logp + _log_add(pb, pnb)),

                    )



        beams = next_beams



    best = max(beams.items(), key=lambda kv: _log_add(kv[1][0], kv[1][1]))

    return list(best[0])





def greedy_decoder(output, text_process, max_frames=None):

    """[T3] Greedy CTC decode."""

    if max_frames is not None:

        output = output[:max_frames]

    arg_maxes = torch.argmax(output, dim=-1)

    decodes = []

    for i in range(len(arg_maxes)):

        if arg_maxes[i] != 0:

            if i == 0 or arg_maxes[i] != arg_maxes[i - 1]:

                decodes.append(arg_maxes[i].item())

    return text_process.int_to_text(decodes)





def decode_frame_logits(frame_logits, text_process, max_frames=None, decoder="greedy", beam_size=5):

    """Decode một mẫu từ logits [T, C] bằng greedy hoặc beam."""

    if max_frames is not None:

        frame_logits = frame_logits[:max_frames]



    if decoder == "beam":

        log_probs = F.log_softmax(frame_logits.float(), dim=-1).cpu()

        tokens = ctc_beam_search(log_probs, beam_width=beam_size, blank=0)

        return text_process.int_to_text(tokens)



    return greedy_decoder(frame_logits, text_process)





def decode_logits(logits, wav_lens, text_process, decoder="greedy", beam_size=5):

    """[T3] Decode cả batch."""

    frame_lens = calculate_input_lengths(wav_lens)

    preds = []

    for idx in range(logits.size(0)):

        t = int(frame_lens[idx].item())

        preds.append(decode_frame_logits(logits[idx, :t], text_process, decoder=decoder, beam_size=beam_size))

    return preds





def evaluate_metrics(model, data_loader, device, max_batches=None, log_samples=3, decoder="greedy", beam_size=5):

    """[T4] WER/CER trên val hoặc test với decoder chỉ định."""

    model.eval()

    total_wer, total_cer, samples = 0.0, 0.0, 0

    logged = 0



    with torch.no_grad():

        for batch_idx, (wavs, labels, w_lens, l_lens) in enumerate(data_loader):

            if max_batches is not None and batch_idx >= max_batches:

                break



            logits, _ = model(wavs.to(device))

            preds = decode_logits(logits, w_lens, text_process, decoder=decoder, beam_size=beam_size)



            for idx, pred in enumerate(preds):

                target = text_process.labels_to_text(labels[idx], l_lens[idx].item())

                if not target:

                    continue

                total_wer += wer(target, pred)

                total_cer += cer(target, pred)

                samples += 1



                if logged < log_samples:

                    dec = decoder.upper()

                    print(f"  [mẫu {logged + 1} | {dec}] Gốc: {target[:80]}")

                    print(f"           Máy: {pred[:80] if pred else '(rỗng)'}")

                    logged += 1



    if samples == 0:

        return 1.0, 1.0, 0

    return total_wer / samples, total_cer / samples, samples



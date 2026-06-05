import sounddevice as sd
from scipy.io.wavfile import write
import time

def record_audio(duration=5, fs=16000, filename="my_voice.wav"):
    print(f"🎤 Bắt đầu ghi âm trong {duration} giây...")
    time.sleep(1) # Chờ 1 giây để chuẩn bị
    print("🔴 ĐANG THU ÂM...")
    
    # Bắt đầu ghi âm
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()  # Đợi cho đến khi thu âm xong
    
    # Lưu file
    write(filename, fs, recording)
    print(f"✅ Đã lưu file: {filename}")

if __name__ == "__main__":
    record_audio()
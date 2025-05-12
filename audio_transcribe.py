import pyaudio
import wave
import whisper

def record_audio(filename="output.wav", duration=5):
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100

    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    print("* recording")

    frames = [stream.read(CHUNK) for _ in range(0, int(RATE / CHUNK * duration))]

    print("* done recording")
    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

def transcribe_audio(filename="output.wav"):
    model = whisper.load_model("base")
    result = model.transcribe(filename)
    return result["text"]

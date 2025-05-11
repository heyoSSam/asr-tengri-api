from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from transformers import pipeline
import io
import soundfile as sf  # Лучше чем librosa для WAV

# Инициализация FastAPI и модели
app = FastAPI()
asr_pipeline = pipeline("automatic-speech-recognition", model="t3ngr1/whisper-small-kk")

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    if not file.filename.endswith(".wav"):
        return JSONResponse(status_code=400, content={"error": "Only .wav files are supported."})

    audio_bytes = await file.read()
    audio_buffer = io.BytesIO(audio_bytes)

    waveform, sample_rate = sf.read(audio_buffer)

    if sample_rate != 16000:
        return JSONResponse(status_code=400, content={"error": "Sample rate must be 16kHz."})

    # Получение результата
    result = asr_pipeline(waveform)
    return {"text": result["text"]}


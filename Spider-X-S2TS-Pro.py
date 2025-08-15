# Spider-X-S2TS-Pro.py
# Spider『X』 — ASR (Whisper) → MT (IndicTrans2) → TTS (IndicF5)
# GPU-safe sequential loading, per-step timing, dark themed UI with animations

import os
import re
import io
import time
import datetime
import tempfile
import warnings
import numpy as np
import soundfile as sf
import torch
import gradio as gr
import librosa
import requests
from tqdm import tqdm
import base64
import os


# Whisper
import whisper

# Translation
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit.processor import IndicProcessor

# TTS
from transformers import AutoModel as AutoTTSModel

warnings.filterwarnings("ignore", message=".*copying from a non-meta parameter.*")

#icon Config

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Construct the full path to the icon file
icon_path = os.path.join(script_dir, "icon.png")

# Read the image file and encode it as Base64
with open(icon_path, "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

# Create the HTML source for the image
# This variable will now hold the Base64 image data
icon_html_src = f"data:image/png;base64,{encoded_string}"

# -----------------------------
# Config & Output Directories
# -----------------------------
ASR_MODEL_SIZE = "large-v3"
ASR_LANGUAGE = "ta"  # Whisper language hint (Tamil default; tweak if needed)

OUTPUT_DIR_TRANSCRIPTS = "Transcriptions"
OUTPUT_DIR_TRANSLATIONS = "Translations"
OUTPUT_DIR_TTS = "Voices"

os.makedirs(OUTPUT_DIR_TRANSCRIPTS, exist_ok=True)
os.makedirs(OUTPUT_DIR_TRANSLATIONS, exist_ok=True)
os.makedirs(OUTPUT_DIR_TTS, exist_ok=True)

# -----------------------------
# Translation Model Config
# -----------------------------
MT_MODEL_NAME = "ai4bharat/indictrans2-indic-indic-1B"
LANG_CODE_MAP = {
    "Assamese": "asm_Beng", "Bengali": "ben_Beng", "Bodo": "brx_Deva", "Dogri": "doi_Deva",
    "English": "eng_Latn", "Gujarati": "guj_Gujr", "Hindi": "hin_Deva", "Kannada": "kan_Knda",
    "Kashmiri (Devanagari)": "kas_Deva", "Kashmiri (Perso-Arabic)": "kas_Arab",
    "Konkani": "gom_Deva", "Maithili": "mai_Deva", "Malayalam": "mal_Mlym",
    "Manipuri (Bengali script)": "mni_Beng", "Manipuri (Meitei script)": "mni_Mtei",
    "Marathi": "mar_Deva", "Nepali": "npi_Deva", "Odia": "ory_Orya", "Punjabi": "pan_Guru",
    "Sanskrit": "san_Deva", "Santali (Devanagari)": "sat_Deva", "Santali (Ol Chiki)": "sat_Olck",
    "Sindhi (Devanagari)": "snd_Deva", "Sindhi (Perso-Arabic)": "snd_Arab",
    "Tamil": "tam_Taml", "Telugu": "tel_Telu", "Urdu": "urd_Arab"
}

GEN_KW = dict(
    use_cache=False,
    min_length=0,
    max_length=256,
    num_beams=5,
    num_return_sequences=1,
)

# -----------------------------
# Device / Precision
# -----------------------------
if torch.cuda.is_available():
    device = "cuda"
    fp16_mode = True
else:
    device = "cpu"
    fp16_mode = False

# -----------------------------
# Helpers
# -----------------------------
_SPLIT_RE = re.compile(r"(?<=[\.\?\!।])\s+|\n+")

def _hhmmss(seconds: float) -> str:
    return time.strftime("%H:%M:%S", time.gmtime(seconds))

def split_into_chunks(text: str, max_chars: int = 600):
    parts = [p.strip() for p in _SPLIT_RE.split(text) if p.strip()]
    chunks, buf, cur_len = [], [], 0
    for p in parts:
        if cur_len + len(p) + 1 > max_chars and buf:
            chunks.append(" ".join(buf))
            buf, cur_len = [p], len(p)
        else:
            buf.append(p)
            cur_len += len(p) + 1
    if buf:
        chunks.append(" ".join(buf))
    if not chunks:
        text = text.strip()
        return [text[i:i+max_chars] for i in range(0, len(text), max_chars)]
    return chunks

def free_vram():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# -----------------------------
# Stage 1 — ASR (GPU-SAFE)
# -----------------------------
def run_asr(audio_file: str, progress: gr.Progress) -> (str, float):
    stage_start = time.time()
    progress(0, desc="[1/3] Loading Whisper…")
    import whisper  # import inside to keep memory footprint small at idle

    model = whisper.load_model(ASR_MODEL_SIZE, device=device)
    progress(0.15, desc="[1/3] Transcribing audio…")
    result = model.transcribe(audio_file, language=ASR_LANGUAGE, fp16=fp16_mode)
    transcription = result.get("text", "")

    # Free ASR from VRAM
    del model
    free_vram()
    progress(0.33, desc=f"[1/3] ASR done in {_hhmmss(time.time() - stage_start)}")

    return transcription, time.time() - stage_start

# -----------------------------
# Stage 2 — MT (GPU-SAFE)
# -----------------------------
def run_translation(text: str, src_lang: str, tgt_lang: str, progress: gr.Progress) -> (str, float):
    stage_start = time.time()
    progress(0.34, desc="[2/3] Loading IndicTrans2…")

    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    from IndicTransToolkit.processor import IndicProcessor

    tokenizer = AutoTokenizer.from_pretrained(MT_MODEL_NAME, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MT_MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)
    model.config.use_cache = False
    iproc = IndicProcessor(inference=True)

    src_code = LANG_CODE_MAP[src_lang]
    tgt_code = LANG_CODE_MAP[tgt_lang]
    chunks = split_into_chunks(text, max_chars=600)

    translated_chunks = []
    BATCH = 8
    total_batches = max(1, (len(chunks) + BATCH - 1) // BATCH)

    for i in range(0, len(chunks), BATCH):
        batch_id = (i // BATCH) + 1
        progress(0.34 + 0.30 * (batch_id / total_batches), desc=f"[2/3] Translating… ({batch_id}/{total_batches})")
        batch = iproc.preprocess_batch(chunks[i:i+BATCH], src_lang=src_code, tgt_lang=tgt_code)
        inputs = tokenizer(batch, truncation=True, padding="longest", return_tensors="pt").to(device)
        with torch.no_grad():
            generated = model.generate(**inputs, **GEN_KW)
        out = tokenizer.batch_decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        out = iproc.postprocess_batch(out, lang=tgt_code)
        translated_chunks.extend(out)

    translation = "\n".join(translated_chunks)

    # Free MT from VRAM
    del model, tokenizer, iproc
    free_vram()
    progress(0.66, desc=f"[2/3] Translation done in {_hhmmss(time.time() - stage_start)}")

    return translation, time.time() - stage_start

# -----------------------------
# Stage 3 — TTS (GPU-SAFE)
# -----------------------------
def run_tts(text: str, ref_audio, ref_text: str, tgt_lang: str, raw_audio_name: str, progress: gr.Progress):
    stage_start = time.time()
    progress(0.67, desc="[3/3] Loading IndicF5…")

    from transformers import AutoModel as AutoTTSModel
    # Guard: need ref audio + reference text for stable voice clone
    if ref_audio is None or (ref_text or "").strip() == "":
        progress(0.99, desc="[3/3] Skipped TTS (missing reference audio/text)")
        return None, None, 0.0

    # Validate ref_audio
    if isinstance(ref_audio, tuple) and len(ref_audio) == 2:
        sample_rate, audio_data = ref_audio
    else:
        progress(0.99, desc="[3/3] Skipped TTS (invalid reference audio format)")
        return None, None, 0.0

    tts_model = AutoTTSModel.from_pretrained("6Morpheus6/IndicF5", trust_remote_code=True).to(device)

    # Save temp prompt for model call
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
        sf.write(temp_audio.name, audio_data, samplerate=sample_rate, format='WAV')

    progress(0.78, desc="[3/3] Generating voice…")
    audio = tts_model(text, ref_audio_path=temp_audio.name, ref_text=ref_text)

    # Normalize to float32 if needed
    if isinstance(audio, np.ndarray) and audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0

    # Persist
    save_name = f"TTS_{tgt_lang}_{os.path.splitext(os.path.basename(raw_audio_name))[0]}.wav"
    save_path = os.path.join(OUTPUT_DIR_TTS, save_name)
    sf.write(save_path, audio, samplerate=24000)

    # Free TTS from VRAM
    del tts_model
    free_vram()
    progress(1.0, desc=f"[3/3] TTS done in {_hhmmss(time.time() - stage_start)}")

    return (24000, audio), save_path, time.time() - stage_start

# -----------------------------
# Full Orchestration
# -----------------------------
def full_pipeline(audio_file, src_lang, tgt_lang, ref_audio, ref_text, progress=gr.Progress()):
    if audio_file is None:
        return (
            "Please upload audio.", None,
            "—", None,
            None, None,
            "00:00:00", "00:00:00", "00:00:00", "00:00:00",
            "No run yet."
        )

    run_start = time.time()
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_audio_name = os.path.basename(audio_file)

    # STEP 1: ASR
    transcription, asr_sec = run_asr(audio_file, progress)
    transcript_path = os.path.join(
        OUTPUT_DIR_TRANSCRIPTS, f"{os.path.splitext(raw_audio_name)[0]}_{ts}.txt"
    )
    with open(transcript_path, "w", encoding="utf-8") as f:
        f.write(transcription)

    # STEP 2: MT
    translation, mt_sec = run_translation(transcription, src_lang, tgt_lang, progress)
    translation_path = os.path.join(
        OUTPUT_DIR_TRANSLATIONS, f"{tgt_lang}_{os.path.splitext(raw_audio_name)[0]}.txt"
    )
    with open(translation_path, "w", encoding="utf-8") as f:
        f.write(translation)

    # STEP 3: TTS
    tts_audio, tts_path, tts_sec = run_tts(translation, ref_audio, ref_text, tgt_lang, raw_audio_name, progress)

    total_sec = time.time() - run_start

    # Build a sleek run log
    log = (
        f"### ✅ Run Summary\n"
        f"- **Audio:** `{raw_audio_name}`\n"
        f"- **ASR (Whisper):** {_hhmmss(asr_sec)}\n"
        f"- **MT (IndicTrans2):** {_hhmmss(mt_sec)}\n"
        f"- **TTS (IndicF5):** {_hhmmss(tts_sec)}\n"
        f"- **Total:** **{_hhmmss(total_sec)}**\n"
        f"\n> Outputs saved in:\n"
        f"- `./{OUTPUT_DIR_TRANSCRIPTS}/`\n"
        f"- `./{OUTPUT_DIR_TRANSLATIONS}/`\n"
        f"- `./{OUTPUT_DIR_TTS}/`\n"
    )

    return (
        transcription, transcript_path,
        translation, translation_path,
        tts_audio, tts_path,
        _hhmmss(asr_sec), _hhmmss(mt_sec), _hhmmss(tts_sec), _hhmmss(total_sec),
        log
    )

# -----------------------------
# Dark Theme + Animated UI
# -----------------------------
# Gradio’s theming API supports hue customization; we’ll add custom CSS for a polished dark look.
dark = gr.themes.Soft(primary_hue="violet", neutral_hue="slate")

CUSTOM_CSS = """
:root, .dark {
  --radius: 16px;
}

.gradio-container {
  max-width: 2200px !important;
}

/* Section card styling */
.section-card {
  border-radius: 16px;
  padding: 16px;
  background: rgba(15, 15, 18, 0.7);
  border: 1px solid rgba(120, 119, 198, 0.25);
  box-shadow: 0 10px 30px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.04);
  transition: transform 0.25s ease, box-shadow 0.25s ease;
  overflow: visible !important; /* prevent clipping */
}

.section-card:hover {
  transform: translateY(-2px);
  box-shadow: 0 14px 36px rgba(0,0,0,0.45);
}

/* Fade-in animation */
.fade-in {
  animation: fadeInUp 420ms ease both;
}

@keyframes fadeInUp {
  0% { opacity: 0; transform: translateY(6px); }
  100% { opacity: 1; transform: translateY(0px); }
}

/* Strong labels */
.label-strong {
  font-weight: 600;
  letter-spacing: 0.2px;
}

.footer-note {
  opacity: 0.7;
  font-size: 12px;
  padding-top: 6px;
}

/* Rounded & larger buttons */
.gradio-container button,
.gradio-container .gr-button,
.gradio-container .gr-button-primary {
  border-radius: 12px !important;
  padding: 10px 14px !important;
}

/* Rounded corners for inputs/outputs */
input, textarea, .wrap .input-component, .wrap .output-component, .gr-box {
  border-radius: 14px !important;
}

/* -------- AUDIO PLAYER FIXES -------- */

/* Remove any clipping so controls are visible */
.section-card .wrap .audio-container,
.section-card .wrap .audio-container > div {
    overflow: visible !important;
}

/* Ensure enough height for full control row */
.section-card .wrap .audio-container {
    min-height: 88px !important;
    height: auto !important;
}

/* Internal wave-player height */
.section-card wave-player {
    height: 100% !important;
    min-height: 88px !important;
}

/* Host container height */
.section-card .wrap .gr-audio {
    min-height: 120px !important;
    height: auto !important;
}

/* Center align main play button row */
.gradio-container .gr-audio .audio-controls {
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}
"""


with gr.Blocks(theme=dark, css=CUSTOM_CSS, fill_height=True) as demo:
    gr.HTML(f"""
    <div class="fade-in" style="display:flex;align-items:center;gap:14px;margin: 6px 2px 16px 2px;">
      <div style="width:50px;height:50px;display:flex;align-items:center;justify-content:center;">
        <img src="{icon_html_src}" style="width:50px;height:50px;"/>
      </div>
      <div>
        <div style="font-size:30px;font-weight:800;">Spider『X』 Speech to Translated Speech Model - 1.0 </div>
        <div style="opacity:0.8">ASR → MT → TTS V-1.0 </div>
      </div>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown("#### Input", elem_classes=["section-card", "fade-in"])
                audio_in = gr.Audio(type="filepath", label="Upload Audio (MP3, WAV, etc.)")
                with gr.Row():
                    src_dd = gr.Dropdown(choices=list(LANG_CODE_MAP.keys()), value="Tamil", label="Source Language")
                    tgt_dd = gr.Dropdown(choices=list(LANG_CODE_MAP.keys()), value="Hindi", label="Target Language")
                ref_audio_in = gr.Audio(type="numpy", label="Reference Voice Audio (for cloning)")
                ref_text_in = gr.Textbox(label="Transcript of Reference Voice Audio", lines=2, placeholder="Enter the text content of the reference voice…")
                run_btn = gr.Button("🚀 Run Full Model", variant="primary")

        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown("#### Progress & Timings", elem_classes=["section-card", "fade-in"])
                time_asr = gr.Label(value="00:00:00", label="ASR Time (HH:MM:SS)")
                time_mt = gr.Label(value="00:00:00", label="MT Time (HH:MM:SS)")
                time_tts = gr.Label(value="00:00:00", label="TTS Time (HH:MM:SS)")
                #time_total = gr.Label(value="00:00:00", label="Total Time (HH:MM:SS)")
                #run_log = gr.Markdown("Run log will appear here.", elem_id="runlog")

            with gr.Group():
                gr.Markdown("#### Total Time & Log", elem_classes=["section-card", "fade-in"])    
                time_total = gr.Label(value="00:00:00", label="Total Time (HH:MM:SS)")
                run_log = gr.Label(value="runlog", label="Run log will appear here.")
            

    with gr.Row():
        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown("#### Transcription", elem_classes=["section-card", "fade-in"])
                transcribed_tb = gr.Textbox(label="Transcribed Text", lines=10)
                transcript_file_out = gr.File(label="Download Transcription (.txt)")
        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown("#### Translation", elem_classes=["section-card", "fade-in"])
                translated_tb = gr.Textbox(label="Translated Text", lines=10)
                translation_file_out = gr.File(label="Download Translation (.txt)")

    with gr.Row():
        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown("#### Voice Output", elem_classes=["section-card", "fade-in"])
                tts_audio_out = gr.Audio(label="Generated Speech", type="numpy")
                tts_file_out = gr.File(label="Download TTS (.wav)")
                gr.Markdown("<div class='footer-note'>Note: (This model was created by Spider-X, Fully working S2TS Model-1.0 ).</div>")

    run_btn.click(
        fn=full_pipeline,
        inputs=[audio_in, src_dd, tgt_dd, ref_audio_in, ref_text_in],
        outputs=[
            transcribed_tb, transcript_file_out,
            translated_tb, translation_file_out,
            tts_audio_out, tts_file_out,
            time_asr, time_mt, time_tts, time_total,
            run_log
        ]
    )

if __name__ == "__main__":
    demo.launch(server_port=7860, inbrowser=False, show_error=True)

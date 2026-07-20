#!/usr/bin/env python3
"""
ai.py — Công cụ lồng tiếng video bằng AI (Việt hóa tự động)
=============================================================
Dịch giọng nói trong video (bất kỳ ngôn ngữ nào) sang tiếng Việt
và thay bằng giọng AI tự nhiên, khớp thời gian với video gốc.

Tương thích Python 3.12 — KHÔNG dùng Coqui/TTS, KHÔNG dùng ffmpeg-python.
Tự động dùng GPU (CUDA) nếu có, không có thì chạy CPU bình thường.

Cách dùng:
    python ai.py input.mp4
    python ai.py input.mp4 --voice male
    python ai.py input.mp4 --output ket_qua.mp4
    python ai.py input.mp4 --model medium        (chính xác hơn, chậm hơn)
    python ai.py input.mp4 --skip_dep_check       (bỏ qua kiểm tra thư viện)
    python ai.py input.mp4 --keep_temp            (giữ file tạm để debug)

Pipeline:
    1. Tách audio    →  FFmpeg (qua subprocess)
    2. Phiên âm      →  Whisper
    3. Dịch thuật    →  Helsinki-NLP/opus-mt (offline, qua tiếng Anh nếu cần)
    4. Tạo giọng AI  →  Microsoft Edge TTS
    5. Căn thời gian →  FFmpeg atempo / thêm im lặng
    6. Ghép audio    →  pydub overlay
    7. Ghép video    →  FFmpeg stream copy
"""

import argparse
import asyncio
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── Cấu hình log ra màn hình ─────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("ai")

# Giọng tiếng Việt Microsoft Edge TTS — miễn phí, tự nhiên, không cần GPU
GIONG_VIET = {
    "female": "vi-VN-HoaiMyNeural",
    "male":   "vi-VN-NamMinhNeural",
}


# ─────────────────────────────────────────────────────────────────────────────
# BƯỚC 1 — TÁCH AUDIO TỪ VIDEO
# ─────────────────────────────────────────────────────────────────────────────

def extract_audio(video_path: str, temp_dir: str) -> str:
    """
    Tách âm thanh từ video bằng FFmpeg, xuất ra WAV 16kHz mono
    (định dạng Whisper xử lý tốt nhất, không phụ thuộc codec gốc).

    Tham số:
        video_path: Đường dẫn file MP4 đầu vào.
        temp_dir:   Thư mục tạm để lưu file trung gian.

    Trả về:
        Đường dẫn file WAV đã tách.
    """
    logger.info(f"Đang tách audio từ: {video_path}")
    audio_path = os.path.join(temp_dir, "audio_goc.wav")

    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn",                    # bỏ luồng video
        "-acodec", "pcm_s16le",   # PCM 16-bit
        "-ar", "16000",           # 16kHz — chuẩn của Whisper
        "-ac", "1",               # mono
        audio_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg tách audio thất bại:\n{result.stderr}")

    logger.info(f"  → {audio_path}")
    return audio_path


# ─────────────────────────────────────────────────────────────────────────────
# BƯỚC 2 — PHIÊN ÂM GIỌNG NÓI (WHISPER)
# ─────────────────────────────────────────────────────────────────────────────

def transcribe(audio_path: str, model_size: str = "small") -> Tuple[List[Dict], str]:
    """
    Phiên âm audio bằng Whisper, tự động nhận diện ngôn ngữ.
    Tự động dùng GPU (CUDA) nếu máy có, không thì chạy CPU.

    Tham số:
        audio_path: Đường dẫn file WAV 16kHz.
        model_size: "tiny" | "base" | "small" | "medium" | "large".

    Trả về:
        (danh_sach_doan, ma_ngon_ngu_phat_hien)
    """
    import torch
    import whisper

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Đang tải model Whisper '{model_size}' trên {device.upper()}…")

    model = whisper.load_model(model_size, device=device)

    logger.info("Đang phiên âm — tự động nhận diện ngôn ngữ…")
    result = model.transcribe(
        audio_path,
        task="transcribe",
        verbose=False,
        word_timestamps=False,
        fp16=(device == "cuda"),
    )

    detected_lang: str = result.get("language", "unknown")
    raw_segments: List[Dict] = result.get("segments", [])

    logger.info(f"  Ngôn ngữ phát hiện : {detected_lang}")
    logger.info(f"  Số đoạn thô        : {len(raw_segments)}")

    # Lọc đoạn rỗng hoặc thời lượng không hợp lệ
    cleaned: List[Dict] = []
    for seg in raw_segments:
        text = seg["text"].strip()
        duration = seg["end"] - seg["start"]
        if not text or duration <= 0:
            continue
        cleaned.append({
            "id":       seg["id"],
            "start":    round(seg["start"], 3),
            "end":      round(seg["end"], 3),
            "duration": round(duration, 3),
            "text":     text,
        })

    logger.info(f"  Số đoạn hợp lệ     : {len(cleaned)}")
    return cleaned, detected_lang


# ─────────────────────────────────────────────────────────────────────────────
# BƯỚC 3 — DỊCH THUẬT SANG TIẾNG VIỆT
# ─────────────────────────────────────────────────────────────────────────────

def _load_translation_pipeline(src_lang: str) -> List[Tuple[str, object, object]]:
    """
    Tải model dịch MarianMT tốt nhất có sẵn (offline, không cần API).

    Chiến lược:
        1. Thử model dịch thẳng: Helsinki-NLP/opus-mt-{src}-vi
        2. Nếu không có → dịch 2 bước: src → tiếng Anh → tiếng Việt
    """
    from transformers import MarianMTModel, MarianTokenizer

    def _load(model_id: str):
        logger.info(f"  Đang tải model: {model_id}")
        tok = MarianTokenizer.from_pretrained(model_id)
        mdl = MarianMTModel.from_pretrained(model_id)
        return tok, mdl

    # Thử dịch thẳng
    direct_id = f"Helsinki-NLP/opus-mt-{src_lang}-vi"
    try:
        tok, mdl = _load(direct_id)
        logger.info("  Đã tải model dịch thẳng.")
        return [("truc_tiep", tok, mdl)]
    except Exception:
        logger.warning(f"  Model '{direct_id}' không có — chuyển sang dịch qua tiếng Anh.")

    # Bước 1: src → tiếng Anh
    pivot_ids = [
        f"Helsinki-NLP/opus-mt-{src_lang}-en",
        f"Helsinki-NLP/opus-mt-tc-big-{src_lang}-en",
    ]
    src_en_tok, src_en_mdl = None, None
    for pid in pivot_ids:
        try:
            src_en_tok, src_en_mdl = _load(pid)
            break
        except Exception:
            continue

    if src_en_tok is None:
        raise RuntimeError(
            f"Không tìm thấy model dịch cho ngôn ngữ '{src_lang}'. "
            "Cần internet để tải model lần đầu."
        )

    # Bước 2: tiếng Anh → tiếng Việt
    en_vi_tok, en_vi_mdl = _load("Helsinki-NLP/opus-mt-en-vi")
    logger.info("  Đã tải pipeline dịch 2 bước (src → en → vi).")
    return [
        ("src_to_en", src_en_tok, src_en_mdl),
        ("en_to_vi",  en_vi_tok,  en_vi_mdl),
    ]


def _translate_batch(texts: List[str], tokenizer, model, batch_size: int = 16) -> List[str]:
    """Dịch danh sách văn bản theo từng lô bằng model MarianMT."""
    results = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        inputs = tokenizer(
            chunk, return_tensors="pt", padding=True,
            truncation=True, max_length=512,
        )
        translated_ids = model.generate(**inputs)
        results.extend([tokenizer.decode(t, skip_special_tokens=True) for t in translated_ids])
    return results


def translate_text(segments: List[Dict], src_lang: str, target_lang: str = "vi") -> List[Dict]:
    """
    Dịch toàn bộ văn bản các đoạn sang tiếng Việt.
    Hoàn toàn offline, không cần API key.
    """
    if src_lang in ("vi", target_lang):
        logger.info("Nguồn đã là tiếng Việt — bỏ qua bước dịch.")
        return segments

    logger.info(f"Đang xây dựng pipeline dịch: {src_lang} → {target_lang}")
    pipeline = _load_translation_pipeline(src_lang)

    current_texts = [seg["text"] for seg in segments]
    for label, tokenizer, model in pipeline:
        logger.info(f"  Đang dịch bước: {label} ({len(current_texts)} đoạn)…")
        current_texts = _translate_batch(current_texts, tokenizer, model)

    translated = []
    for seg, vi_text in zip(segments, current_texts):
        s = dict(seg)
        s["original_text"] = seg["text"]
        s["text"] = vi_text.strip()
        translated.append(s)

    logger.info("Dịch thuật hoàn tất.")
    return translated


# ─────────────────────────────────────────────────────────────────────────────
# BƯỚC 4 — TẠO GIỌNG AI TIẾNG VIỆT (MICROSOFT EDGE TTS)
# ─────────────────────────────────────────────────────────────────────────────

_xtts_model = None  # cache model để không tải lại mỗi lần gọi


def _get_xtts_model():
    """Tải model XTTS-v2 (chỉ 1 lần, dùng lại cho toàn bộ pipeline).

    ⚠️ Model XTTS-v2 dùng giấy phép Coqui Public Model License (CPML) —
    CHỈ cho phép sử dụng phi thương mại (cá nhân/nghiên cứu/hobby).
    Xem: https://coqui.ai/cpml
    """
    global _xtts_model
    if _xtts_model is None:
        # Tự động chấp nhận CPML để không bị treo chờ nhập "y" trên server
        # không tương tác (điều này KHÔNG thay đổi nội dung giấy phép, chỉ
        # ghi nhận rằng bạn đã đọc và đồng ý với CPML trước khi model tải).
        os.environ["COQUI_TOS_AGREED"] = "1"
        import torch
        from TTS.api import TTS
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"  Đang tải model XTTS-v2 trên {device.upper()} (lần đầu sẽ hơi lâu)…")
        _xtts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    return _xtts_model


def generate_tts(
    segments: List[Dict],
    temp_dir: str,
    voice: str = "female",
    voice_sample: Optional[str] = None,
) -> List[Dict]:
    """
    Tạo file âm thanh tiếng Việt cho từng đoạn.

    - Nếu có `voice_sample` (đường dẫn file audio mẫu ~6-30 giây): dùng XTTS-v2
      để NHÂN BẢN giọng nói trong file mẫu đó (voice cloning).
    - Nếu không: dùng Microsoft Edge TTS với giọng có sẵn (nam/nữ).

    ⚠️ Lưu ý trách nhiệm: chỉ nhân bản giọng nói của chính bạn hoặc người đã
    đồng ý cho phép — không dùng để giả mạo người khác.

    Đoạn lỗi sẽ tự động chèn im lặng thay vì làm crash cả pipeline.
    """
    from pydub import AudioSegment as PydubAudio

    tts_dir = os.path.join(temp_dir, "doan_tts")
    os.makedirs(tts_dir, exist_ok=True)

    use_cloning = bool(voice_sample) and os.path.isfile(voice_sample)

    if use_cloning:
        logger.info(f"  Chế độ: NHÂN BẢN GIỌNG NÓI từ file mẫu: {voice_sample}")
        xtts = _get_xtts_model()
    else:
        import edge_tts
        voice_name = GIONG_VIET.get(voice, GIONG_VIET["female"])
        logger.info(f"  Chế độ: giọng Edge TTS có sẵn ({voice_name})")

        async def _synthesize(text: str, mp3_path: str) -> None:
            communicate = edge_tts.Communicate(text, voice_name)
            await communicate.save(mp3_path)

    result: List[Dict] = []
    total = len(segments)

    for idx, seg in enumerate(segments):
        logger.info(
            f"  TTS [{idx + 1:>3}/{total}] ({seg['start']:.1f}s): "
            f"{seg['text'][:70]}{'…' if len(seg['text']) > 70 else ''}"
        )

        wav_path = os.path.join(tts_dir, f"doan_{idx:04d}.wav")
        s = dict(seg)

        try:
            if use_cloning:
                xtts.tts_to_file(
                    text=seg["text"],
                    speaker_wav=voice_sample,
                    language="vi",
                    file_path=wav_path,
                )
                audio = PydubAudio.from_wav(wav_path)
            else:
                mp3_path = os.path.join(tts_dir, f"doan_{idx:04d}.mp3")
                asyncio.run(_synthesize(seg["text"], mp3_path))
                audio = PydubAudio.from_mp3(mp3_path)
                audio.export(wav_path, format="wav")

            s["tts_path"] = wav_path
            s["tts_duration"] = len(audio) / 1000.0

        except Exception as exc:
            logger.warning(f"  TTS thất bại đoạn {idx} — chèn im lặng. Lỗi: {exc}")
            silence = PydubAudio.silent(duration=int(seg["duration"] * 1000))
            silence.export(wav_path, format="wav")
            s["tts_path"] = wav_path
            s["tts_duration"] = seg["duration"]

        result.append(s)

    logger.info("Tạo giọng AI hoàn tất.")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# BƯỚC 5 — CĂN THỜI GIAN TỪNG ĐOẠN
# ─────────────────────────────────────────────────────────────────────────────

def _build_atempo_chain(speed_factor: float) -> List[str]:
    """
    Tạo chuỗi filter atempo cho FFmpeg.
    FFmpeg giới hạn atempo trong [0.5, 2.0] nên phải ghép nhiều tầng
    khi cần tăng/giảm tốc nhiều hơn khoảng đó.
    """
    filters: List[str] = []
    remaining = speed_factor

    while remaining > 2.0:
        filters.append("atempo=2.0")
        remaining /= 2.0

    while remaining < 0.5:
        filters.append("atempo=0.5")
        remaining *= 2.0

    if abs(remaining - 1.0) > 0.005:
        filters.append(f"atempo={remaining:.6f}")

    return filters or ["atempo=1.0"]


def adjust_speed(segment: Dict, temp_dir: str, index: int) -> str:
    """
    Căn chỉnh thời lượng đoạn TTS khớp với thời lượng gốc.

    Logic:
        ratio = thoi_luong_goc / thoi_luong_tts

        ≈ 1.0  → copy nguyên (sai lệch < 5%)
        > 1.0  → TTS ngắn hơn → thêm im lặng vào cuối
        < 1.0  → TTS dài hơn  → tăng tốc bằng FFmpeg atempo
                 rồi cắt/đệm chính xác để không lệch tích lũy
    """
    from pydub import AudioSegment as PydubAudio

    orig_dur = segment["duration"]
    tts_path = segment["tts_path"]
    tts_dur = segment["tts_duration"]

    aligned_dir = os.path.join(temp_dir, "doan_can_chinh")
    os.makedirs(aligned_dir, exist_ok=True)
    out_path = os.path.join(aligned_dir, f"can_chinh_{index:04d}.wav")

    # Trường hợp đặc biệt: đoạn quá ngắn
    if orig_dur < 0.1:
        PydubAudio.silent(duration=max(100, int(orig_dur * 1000))).export(out_path, format="wav")
        return out_path

    if tts_dur < 0.05:
        PydubAudio.silent(duration=int(orig_dur * 1000)).export(out_path, format="wav")
        return out_path

    ratio = orig_dur / tts_dur
    target_ms = int(orig_dur * 1000)

    if 0.95 <= ratio <= 1.05:
        # Sai lệch < 5% — copy thẳng, không cần xử lý
        shutil.copy(tts_path, out_path)

    elif ratio > 1.0:
        # TTS ngắn hơn gốc → thêm im lặng vào cuối
        audio = PydubAudio.from_wav(tts_path)
        pad_ms = target_ms - len(audio)
        if pad_ms > 0:
            audio = audio + PydubAudio.silent(duration=pad_ms)
        audio[:target_ms].export(out_path, format="wav")

    else:
        # TTS dài hơn gốc → tăng tốc bằng atempo
        speed_factor = tts_dur / orig_dur
        filter_chain = ",".join(_build_atempo_chain(speed_factor))

        cmd = ["ffmpeg", "-y", "-i", tts_path, "-filter:a", filter_chain, out_path]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            logger.warning(f"  atempo thất bại đoạn {index} — dùng file gốc.")
            shutil.copy(tts_path, out_path)

        # Cắt/đệm để đảm bảo đúng thời lượng tuyệt đối (chống lệch tích lũy)
        audio = PydubAudio.from_wav(out_path)
        if len(audio) > target_ms:
            audio = audio[:target_ms]
        elif len(audio) < target_ms:
            audio = audio + PydubAudio.silent(duration=target_ms - len(audio))
        audio.export(out_path, format="wav")

    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# BƯỚC 6 — GHÉP AUDIO ĐẦY ĐỦ
# ─────────────────────────────────────────────────────────────────────────────

def assemble_audio(
    segments: List[Dict],
    aligned_paths: List[str],
    original_audio_path: str,
    temp_dir: str,
) -> str:
    """
    Ghép từng đoạn đã căn chỉnh vào đúng vị trí timestamp trên track audio.
    Dùng canvas im lặng dài bằng video gốc, overlay từng đoạn lên trên.
    """
    from pydub import AudioSegment as PydubAudio

    logger.info("Đang ghép toàn bộ track audio…")

    original = PydubAudio.from_wav(original_audio_path)
    total_ms = len(original)

    full_track = PydubAudio.silent(duration=total_ms)

    for seg, aligned_path in zip(segments, aligned_paths):
        start_ms = int(seg["start"] * 1000)

        try:
            seg_audio = PydubAudio.from_wav(aligned_path)
        except Exception as exc:
            logger.warning(f"  Bỏ qua đoạn (không đọc được file): {exc}")
            continue

        if start_ms >= total_ms:
            logger.warning(f"  Đoạn tại {seg['start']:.1f}s vượt cuối video — bỏ qua.")
            continue

        max_seg_ms = total_ms - start_ms
        if len(seg_audio) > max_seg_ms:
            seg_audio = seg_audio[:max_seg_ms]

        full_track = full_track.overlay(seg_audio, position=start_ms)

    full_track = full_track.normalize()

    out_path = os.path.join(temp_dir, "audio_da_ghep.wav")
    full_track.export(out_path, format="wav")

    logger.info(f"  Đã ghép: {out_path}  ({total_ms / 1000:.1f}s)")
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# BƯỚC 7 — GHÉP AUDIO VÀO VIDEO
# ─────────────────────────────────────────────────────────────────────────────

def merge_video(video_path: str, audio_path: str, output_path: str) -> str:
    """
    Thay track audio trong video gốc bằng audio tiếng Việt mới.
    Video được stream-copy — không re-encode, nhanh, giữ nguyên chất lượng.
    """
    logger.info(f"Đang ghép audio vào video → {output_path}")

    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", audio_path,
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-shortest",
        output_path,
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"FFmpeg ghép video thất bại:\n{proc.stderr}")

    logger.info(f"  Đầu ra: {output_path}")
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# TIỆN ÍCH — TIỀN XỬ LÝ ĐOẠN
# ─────────────────────────────────────────────────────────────────────────────

def _split_long_segment(seg: Dict, max_duration: float = 10.0) -> List[Dict]:
    """Tách đoạn quá dài, ưu tiên: dấu câu → dấu phẩy → số từ."""
    if seg["duration"] <= max_duration:
        return [seg]

    text = seg["text"]
    sentences = re.split(r"(?<=[.!?।。！？])\s+", text)

    if len(sentences) == 1:
        sentences = [s.strip() for s in text.split(",") if s.strip()]

    if len(sentences) == 1:
        words = text.split()
        mid = len(words) // 2
        sentences = [" ".join(words[:mid]), " ".join(words[mid:])]

    n_chunks = max(2, int(seg["duration"] / max_duration) + 1)
    chunk_size = max(1, len(sentences) // n_chunks)
    chunks = [" ".join(sentences[i : i + chunk_size]) for i in range(0, len(sentences), chunk_size)]
    chunks = [c for c in chunks if c.strip()]

    if not chunks:
        return [seg]

    chunk_dur = seg["duration"] / len(chunks)
    result = []
    for i, chunk_text in enumerate(chunks):
        start = seg["start"] + i * chunk_dur
        result.append({
            "id":            f"{seg['id']}_{i}",
            "start":         round(start, 3),
            "end":           round(start + chunk_dur, 3),
            "duration":      round(chunk_dur, 3),
            "text":          chunk_text,
            "original_text": seg.get("original_text", chunk_text),
        })
    return result


def preprocess_segments(
    segments: List[Dict],
    min_duration: float = 0.35,
    max_duration: float = 10.0,
) -> List[Dict]:
    """Bỏ đoạn quá ngắn (nhiễu), tách đoạn quá dài thành đoạn nhỏ hơn."""
    processed: List[Dict] = []
    short_count = split_count = 0

    for seg in segments:
        if seg["duration"] < min_duration:
            short_count += 1
        elif seg["duration"] > max_duration:
            processed.extend(_split_long_segment(seg, max_duration))
            split_count += 1
        else:
            processed.append(seg)

    logger.info(
        f"  Tiền xử lý: bỏ {short_count} đoạn ngắn, "
        f"tách {split_count} đoạn dài → tổng {len(processed)} đoạn"
    )
    return processed


# ─────────────────────────────────────────────────────────────────────────────
# TIỆN ÍCH — KIỂM TRA DEPENDENCIES
# ─────────────────────────────────────────────────────────────────────────────

def check_dependencies(need_voice_cloning: bool = False) -> None:
    """Kiểm tra FFmpeg và toàn bộ thư viện Python cần thiết.
    Nếu need_voice_cloning=True, kiểm tra thêm package 'TTS' (Coqui, dùng cho XTTS-v2)."""
    if subprocess.run(["ffmpeg", "-version"], capture_output=True).returncode != 0:
        raise RuntimeError(
            "Không tìm thấy FFmpeg.\n"
            "  Cài đặt: sudo apt update && sudo apt install -y ffmpeg"
        )

    required = {
        "whisper":       "openai-whisper",
        "edge_tts":      "edge-tts",
        "transformers":  "transformers",
        "pydub":         "pydub",
        "torch":         "torch",
        "sentencepiece": "sentencepiece",
        "sacremoses":    "sacremoses",
    }
    if need_voice_cloning:
        required["TTS"] = "coqui-tts"

    missing = []
    for module, pip_name in required.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(pip_name)

    if missing:
        raise RuntimeError(
            f"Thiếu thư viện: {', '.join(missing)}\n"
            f"  Cài đặt: pip install {' '.join(missing)}"
        )

    logger.info("Tất cả dependencies đã sẵn sàng.")


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE CHÍNH
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    input_video: str,
    target_lang: str = "vi",
    voice: str = "female",
    model_size: str = "small",
    output_path: Optional[str] = None,
    keep_temp: bool = False,
    voice_sample: Optional[str] = None,
) -> str:
    """Điều phối toàn bộ pipeline lồng tiếng từ đầu đến cuối."""
    wall_start = time.time()

    if not os.path.isfile(input_video):
        raise FileNotFoundError(f"Không tìm thấy file video: {input_video}")

    if output_path is None:
        stem = Path(input_video).stem
        output_path = str(Path(input_video).parent / f"{stem}_vi.mp4")

    temp_dir = tempfile.mkdtemp(prefix="ai_dubbing_")
    logger.info(f"Thư mục tạm: {temp_dir}")

    def _banner(step: int, title: str) -> None:
        logger.info("")
        logger.info(f"{'─' * 60}")
        logger.info(f"  BƯỚC {step}: {title}")
        logger.info(f"{'─' * 60}")

    try:
        _banner(1, "Tách audio từ video")
        audio_path = extract_audio(input_video, temp_dir)

        _banner(2, f"Phiên âm bằng Whisper ({model_size})")
        segments, detected_lang = transcribe(audio_path, model_size)

        if not segments:
            raise RuntimeError(
                "Không phát hiện giọng nói trong video. "
                "Đảm bảo video có âm thanh rõ ràng."
            )

        _save_json(
            {"language": detected_lang, "segments": segments},
            os.path.join(temp_dir, "phien_am.json"),
        )

        segments = preprocess_segments(segments)

        _banner(3, f"Dịch thuật  {detected_lang} → {target_lang}")
        segments = translate_text(segments, detected_lang, target_lang)
        _save_json(segments, os.path.join(temp_dir, "ban_dich.json"))

        _banner(4, "Tạo giọng AI tiếng Việt")
        segments = generate_tts(segments, temp_dir, voice=voice, voice_sample=voice_sample)

        _banner(5, "Căn chỉnh thời gian từng đoạn")
        aligned_paths: List[str] = []
        for idx, seg in enumerate(segments):
            aligned_paths.append(adjust_speed(seg, temp_dir, idx))

        _banner(6, "Ghép toàn bộ track audio")
        assembled = assemble_audio(segments, aligned_paths, audio_path, temp_dir)

        _banner(7, "Ghép audio vào video cuối cùng")
        merge_video(input_video, assembled, output_path)

    finally:
        if keep_temp:
            logger.info(f"\nFile tạm giữ lại tại: {temp_dir}")
        else:
            shutil.rmtree(temp_dir, ignore_errors=True)
            logger.info("Đã xóa file tạm.")

    elapsed = time.time() - wall_start
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"  ✓  Hoàn tất sau {elapsed:.1f} giây")
    logger.info(f"  ✓  Video đầu ra: {output_path}")
    logger.info("=" * 60)
    return output_path


def _save_json(data, path: str) -> None:
    """Ghi dữ liệu ra file JSON UTF-8 (bỏ qua lỗi ghi)."""
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="ai.py",
        description="Lồng tiếng video AI — dịch giọng nói sang tiếng Việt tự nhiên.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
  python ai.py input.mp4                     # giọng nữ, tự đặt tên file ra
  python ai.py input.mp4 --voice male         # giọng nam
  python ai.py input.mp4 --output out.mp4     # đặt tên file ra
  python ai.py input.mp4 --model medium       # chính xác hơn, chậm hơn
  python ai.py input.mp4 --keep_temp          # giữ file tạm để debug
  python ai.py input.mp4 --skip_dep_check     # bỏ qua kiểm tra thư viện
        """,
    )
    parser.add_argument("input", metavar="FILE_VIDEO", help="Đường dẫn file MP4 đầu vào.")
    parser.add_argument("--target_lang", default="vi", help="Ngôn ngữ đích (mặc định: vi).")
    parser.add_argument("--voice", choices=["female", "male"], default="female", help="Giọng AI.")
    parser.add_argument(
        "--voice_sample", default=None,
        help="Đường dẫn file audio mẫu (~6-30s) để NHÂN BẢN giọng nói thay vì dùng giọng có sẵn. "
             "⚠️ Chỉ dùng giọng của chính bạn hoặc người đã đồng ý.",
    )
    parser.add_argument(
        "--model",
        choices=["tiny", "base", "small", "medium", "large"],
        default="small",
        help="Model Whisper (mặc định: small — nhẹ, ổn định trên CPU).",
    )
    parser.add_argument("--output", default=None, help="File MP4 đầu ra.")
    parser.add_argument("--keep_temp", action="store_true", help="Giữ lại thư mục tạm.")
    parser.add_argument("--skip_dep_check", action="store_true", help="Bỏ qua kiểm tra thư viện.")

    args = parser.parse_args()

    if not args.skip_dep_check:
        try:
            check_dependencies(need_voice_cloning=bool(args.voice_sample))
        except (RuntimeError, FileNotFoundError) as exc:
            logger.error(f"Kiểm tra dependency thất bại:\n{exc}")
            sys.exit(1)

    try:
        run_pipeline(
            input_video=args.input,
            target_lang=args.target_lang,
            voice=args.voice,
            model_size=args.model,
            output_path=args.output,
            keep_temp=args.keep_temp,
            voice_sample=args.voice_sample,
        )
    except FileNotFoundError as exc:
        logger.error(str(exc))
        sys.exit(1)
    except RuntimeError as exc:
        logger.error(f"Lỗi pipeline: {exc}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.warning("Người dùng dừng chương trình.")
        sys.exit(130)


if __name__ == "__main__":
    main()

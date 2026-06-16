"""
inference_engines/audio_engine.py
Audio Demuxing + Whisper ASR + Synthetic Voice Classification
"""

import os
import subprocess
import torch
import torch.nn as nn
import numpy as np
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from transformers import (
    WhisperModel,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

from typing import Dict, Any, Optional

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_NAME = "openai/whisper-tiny"

SAMPLE_RATE = 16000
MAX_SECONDS = 10
MAX_SAMPLES = SAMPLE_RATE * MAX_SECONDS


class WhisperEncoder(nn.Module):

    def __init__(self, model_name=MODEL_NAME):
        super().__init__()

        whisper = WhisperModel.from_pretrained(model_name)

        self.encoder = whisper.encoder
        self.hidden_size = whisper.config.d_model

        for i, layer in enumerate(self.encoder.layers):
            if i < len(self.encoder.layers) - 2:
                for p in layer.parameters():
                    p.requires_grad = False

    def forward(self, x):
        out = self.encoder(x)
        pooled = out.last_hidden_state.mean(dim=1)
        return pooled


class AudioDeepfakeClassifier(nn.Module):

    def __init__(self, input_dim, num_classes=1):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)


def extract_audio(video_path, output_wav_path):

    if os.path.exists(output_wav_path):
        try:
            os.remove(output_wav_path)
        except:
            pass

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(SAMPLE_RATE),
        "-ac",
        "1",
        output_wav_path,
    ]

    try:
        subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )

        return (
            os.path.exists(output_wav_path)
            and os.path.getsize(output_wav_path) > 0
        )

    except Exception:
        return False


class AudioAnalyzer:

    def __init__(self, checkpoint_path):

        self.feature_extractor = None
        self.encoder = None
        self.classifier = None
        self.asr_model = None
        self.processor = None

        try:
            self.feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_NAME)

            self.encoder = WhisperEncoder(MODEL_NAME)
            self.encoder.eval().to(DEVICE)

            # Full Whisper model (encoder+decoder) for actual transcription.
            # The WhisperEncoder above only pools hidden states for the
            # deepfake classifier — it never decodes text, which is why
            # transcripts were always empty.
            self.asr_model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
            self.asr_model.eval().to(DEVICE)

            self.processor = WhisperProcessor.from_pretrained(MODEL_NAME)

            self.classifier = AudioDeepfakeClassifier(
                input_dim=self.encoder.hidden_size
            )

            #
            # Compatible with app.py
            #

            if os.path.isfile(checkpoint_path):
                ckpt_file = checkpoint_path

            elif os.path.exists(os.path.join(checkpoint_path, "checkpoint.pth")):
                ckpt_file = os.path.join(checkpoint_path, "checkpoint.pth")

            else:
                ckpt_file = os.path.join(checkpoint_path, "audio_classifier.pth")

            if os.path.exists(ckpt_file):
                state_dict = torch.load(ckpt_file, map_location=DEVICE)
                
                if any(k.startswith("encoder.") or k.startswith("classifier.") for k in state_dict.keys()):
                    encoder_dict = {k.replace("encoder.", ""): v for k, v in state_dict.items() if k.startswith("encoder.")}
                    classifier_dict = {k.replace("classifier.", ""): v for k, v in state_dict.items() if k.startswith("classifier.")}
                    
                    if encoder_dict and hasattr(self.encoder, "load_state_dict"):
                        self.encoder.load_state_dict(encoder_dict, strict=False)
                    self.classifier.load_state_dict(classifier_dict)
                else:
                    self.classifier.load_state_dict(state_dict)

            self.classifier.eval().to(DEVICE)

        except Exception as e:
            print("Audio init error:", e)

            self.feature_extractor = None
            self.encoder = None
            self.classifier = None
            self.asr_model = None
            self.processor = None

    def _load_audio(self, audio_path):

        y, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
        y = y.astype(np.float32)

        if len(y) >= MAX_SAMPLES:
            y = y[:MAX_SAMPLES]
        else:
            y = np.pad(y, (0, MAX_SAMPLES - len(y)))

        return y

    def _transcribe(self, audio_path):
        """
        Full Arabic transcription via Whisper's encoder-decoder generate().
        Note: this re-loads the audio independently of _load_audio() (which
        truncates/pads to MAX_SAMPLES for the classifier) so transcripts
        aren't limited to the 10s classifier window.
        """
        if self.asr_model is None or self.processor is None:
            return ""

        try:
            y, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
            y = y.astype(np.float32)

            if len(y) == 0:
                return ""

            inputs = self.processor(
                y,
                sampling_rate=SAMPLE_RATE,
                return_tensors="pt"
            ).input_features.to(DEVICE)

            with torch.no_grad():
                predicted_ids = self.asr_model.generate(
                    inputs,
                    language="ar",
                    task="transcribe",
                    max_new_tokens=444,
                )

            text = self.processor.batch_decode(
                predicted_ids.cpu(), skip_special_tokens=True
            )[0]

            return text.strip()

        except Exception as e:
            print("Transcription error:", e)
            return ""

    def _compute_saliency(self, feat):

        try:
            feat = feat.clone().detach().requires_grad_(True)

            self.encoder.zero_grad()
            self.classifier.zero_grad()

            emb = self.encoder(feat)
            logit = self.classifier(emb)
            logit.backward()

            grads = feat.grad.detach().cpu().numpy()[0]
            inp = feat.detach().cpu().numpy()[0]

            attr = np.abs(grads * inp)
            denom = attr.max() - attr.min()
            if denom < 1e-5:
                return np.zeros_like(attr)
            attr = (attr - attr.min()) / denom

            return attr

        except:
            return np.zeros(feat.cpu().numpy()[0].shape)

    def _locate_manipulation_windows(self, feat):
        # feat shape: [mel_bins, time_frames]
        # Compute variance across mel_bins for each time frame → [time_frames]
        variances = (
            torch.var(feat, dim=0)
            .cpu()
            .numpy()
            .flatten()
        )

        n_frames = len(variances)

        idx = np.where(
            variances > np.mean(variances) + 2 * np.std(variances)
        )[0]

        if len(idx) == 0:
            return (
                "No structural phase shifts or "
                "pitch variance anomalies detected."
            )

        start = round(idx[0] * MAX_SECONDS / max(n_frames - 1, 1), 1)
        end = round(idx[-1] * MAX_SECONDS / max(n_frames - 1, 1), 1)

        def _fmt(sec):
            m, s = divmod(sec, 60)
            return f"{int(m):02d}:{s:04.1f}"

        return (
            f"Suspicious frame variance window "
            f"localized between {_fmt(start)}"
            f" - {_fmt(end)}"
        )

    def _peak_suspicion_time(self, attr):

        curve = attr.mean(axis=0)
        peak = np.argmax(curve)

        return round(peak * MAX_SECONDS / max(len(curve) - 1, 1), 2)

    def _suspicious_freq_bands(self, attr, threshold=0.35):
        band_mean = attr.mean(axis=1)   # [mel_bins]

        hot = np.where(band_mean > threshold)[0]

        if len(hot) == 0:
            return None

        n_mels = attr.shape[0]
        mel_freqs = librosa.mel_frequencies(
            n_mels=n_mels, fmin=0, fmax=SAMPLE_RATE // 2
        )

        ranges, start = [], hot[0]
        for i in range(1, len(hot)):
            if hot[i] != hot[i - 1] + 1:
                f_lo = int(mel_freqs[start])
                f_hi = int(mel_freqs[hot[i - 1]])
                ranges.append(f"{f_lo}–{f_hi} Hz")
                start = hot[i]
        f_lo = int(mel_freqs[start])
        f_hi = int(mel_freqs[hot[-1]])
        ranges.append(f"{f_lo}–{f_hi} Hz")

        return ", ".join(ranges[:4])

    def analyze(self, audio_path):

        if (
            not audio_path
            or not os.path.exists(audio_path)
            or os.path.getsize(audio_path) == 0
        ):
            return {
                "score": 1.0,
                "label": "Real",
                "confidence": 0.95,
                "prob_real": 1.0,
                "prob_fake": 0.0,
                "transcript": "No audio channel track identified.",
                "manipulated_segments": "None detected",
                "peak_suspicion_time": None,
                "suspicious_freq_bands": None,
                "result_plot": None
            }

        if (
            self.feature_extractor is None
            or self.encoder is None
            or self.classifier is None
        ):
            return {
                "score": 0.5,
                "label": "Real",
                "confidence": 0.5,
                "prob_real": 0.5,
                "prob_fake": 0.5,
                "transcript": "Audio model unavailable.",
                "manipulated_segments": "Unable to compute.",
                "peak_suspicion_time": None,
                "suspicious_freq_bands": None,
                "result_plot": None
            }

        try:
            raw_audio = self._load_audio(audio_path)

            inputs = self.feature_extractor(
                raw_audio,
                sampling_rate=SAMPLE_RATE,
                return_tensors="pt",
                chunk_length=MAX_SECONDS
            )

            feat = inputs.input_features.to(DEVICE)

            with torch.no_grad():
                emb = self.encoder(feat)
                logit = self.classifier(emb)
                prob_fake = float(torch.sigmoid(logit))

            prob_real = 1.0 - prob_fake

            label = "Fake" if prob_fake >= 0.5 else "Real"

            confidence = prob_fake if label == "Fake" else prob_real

            try:
                attr = self._compute_saliency(feat)
                peak_time = self._peak_suspicion_time(attr)
                freq_bands = self._suspicious_freq_bands(attr)
            except:
                attr = np.zeros(feat.cpu().numpy()[0].shape)
                peak_time = None
                freq_bands = None

            try:
                manip = (
                    self._locate_manipulation_windows(feat[0])
                    if label == "Fake"
                    else "No structural phase shifts or pitch variance anomalies detected."
                )
            except:
                manip = "Analysis aborted on sub-segment isolation step."

            # Actual decoded transcript (was hardcoded to "" before)
            transcript = self._transcribe(audio_path)

            return {
                "score": prob_fake,
                "label": label,
                "confidence": confidence,
                "prob_real": prob_real,
                "prob_fake": prob_fake,
                "transcript": transcript,
                "manipulated_segments": manip,
                "peak_suspicion_time": peak_time,
                "suspicious_freq_bands": freq_bands,
                "result_plot": None
            }

        except Exception as e:
            print("Audio analyze error:", e)

            return {
                "score": 0.5,
                "label": "Error",
                "confidence": 0.5,
                "prob_real": 0.5,
                "prob_fake": 0.5,
                "transcript": "Audio execution runtime failure.",
                "manipulated_segments": "Analysis aborted.",
                "peak_suspicion_time": None,
                "suspicious_freq_bands": None,
                "result_plot": None
            }
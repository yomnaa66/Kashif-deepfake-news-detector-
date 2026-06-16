"""
pipeline_orchestrator.py

Bug fixes applied (v2):
  [P1] tmp_dir now cleaned up even when audio extraction fails
  [P2] ThreadPoolExecutor block wrapped in try/except — results['error'] populated
  [P3] Empty audio path short-circuited before calling AudioAnalyzer.analyze()
  [Q4] video_path validated (exists + readable) before any work begins

Bug fixes applied (v3) — reconciling against actual engine contracts:
  [R1] Audio "no track" / "unavailable" / "runtime failure" placeholder transcripts
       are now filtered out before being handed to TextAnalyzer.analyze(), instead
       of being treated as real speech (they were long enough to skip the text
       engine's own empty-string guard, so it was running real classification on
       placeholder sentences and returning misleading Real/Fake verdicts).
  [R2] VideoAnalyzer.process_video() has no internal try/except (unlike audio/text,
       which always return a dict, never raise). A crash inside it now produces a
       structured video result with label='Error' instead of leaving results['video']
       empty while results['error'] is set, so all three modalities expose failures
       the same way to callers (e.g. app.py).
  [R3] VideoAnalyzer(...) construction (in __init__) can raise (no internal
       try/except, unlike AudioAnalyzer/TextAnalyzer which degrade to model=None).
       Wrapped so the whole pipeline object still constructs, with video_analyzer
       set to None and process_video calls short-circuited to an Error result.

Bug fixes applied (v4) — race condition between transcription and cleanup:
  [S1] _cleanup() was placed in the outer finally block, which fires as soon as
       the inner try/except exits — including via the early `return results` inside
       the except branch. This meant audio_wav could be deleted while
       _transcribe_audio() was still mid-read inside librosa.load(), causing a
       FileNotFoundError that silently returned "" and sent an empty transcript to
       TextAnalyzer (which then returned a neutral 0.5/0.5 result with no real
       inference). Fix: _cleanup() is now called explicitly at the end of the
       with-block (after all futures are resolved) AND kept in finally as a
       safety net for the error-exit path — but the file is already gone by then
       so _cleanup() no-ops safely on the missing file.
  [S2] _transcribe_audio() passed the full raw numpy array to asr_processor(),
       which internally truncates to a single 30-second mel window. For videos
       longer than 30s (e.g. 46s), all speech after second 30 was silently
       discarded. If the first 30s is a music intro with no clear speech, Whisper
       returns "" or ".", which collapses to 0 tokens in preprocess_arabic and
       causes TextAnalyzer to return a neutral 0.5/0.5 result. Fix: audio is now
       chunked into overlapping 25s windows (5s overlap, 20s step) and each chunk
       is transcribed independently — covering the full audio regardless of length.
"""


import os
import tempfile
import concurrent.futures
from typing import Dict, Any

import numpy as np
import librosa
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

from inference_engines.video_engine import VideoAnalyzer
from inference_engines.audio_engine import AudioAnalyzer, extract_audio, SAMPLE_RATE, MODEL_NAME, DEVICE
from inference_engines.text_engine  import TextAnalyzer


class MultimodalForensicPipeline:
    def __init__(
        self,
        video_model_path: str = "model_weights/deepfake_vision/best_model.pth",
        audio_model_path: str = "model_weights/whisper_audio/checkpoint.pth",
        text_model_path:  str = "model_weights/arabert_news/",
    ):
        # [R3] FIX: VideoAnalyzer has no internal try/except on init (unlike
        #      AudioAnalyzer/TextAnalyzer, which degrade to model=None on failure).
        #      Guard it here so a bad checkpoint/MTCNN init doesn't prevent the
        #      whole pipeline object from constructing.
        try:
            self.video_analyzer = VideoAnalyzer(model_path=video_model_path)
        except Exception as e:
            print(f"[MultimodalForensicPipeline] VideoAnalyzer init failed: {e}")
            self.video_analyzer = None

        self.audio_analyzer = AudioAnalyzer(checkpoint_path=audio_model_path)
        self.text_analyzer  = TextAnalyzer(model_dir=text_model_path)

        # Whisper ASR — loaded here so audio_engine only handles classification
        try:
            self._asr_processor = WhisperProcessor.from_pretrained(MODEL_NAME)
            self._asr_model     = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
            self._asr_model.eval().to(DEVICE)
        except Exception as e:
            print(f'[Pipeline] Whisper ASR init failed: {e}')
            self._asr_processor = None
            self._asr_model     = None


    # ── Transcription ──────────────────────────────────────────────────────────
    def _transcribe_audio(self, audio_wav: str) -> str:
        """
        Transcribe the full audio file using Whisper with chunked long-form decoding.

        [S2] FIX: the previous implementation passed the raw numpy array directly
        to asr_processor(), which internally truncates to a single 30-second
        mel-spectrogram window. For videos longer than 30s (e.g. 46s), any speech
        after the 30-second mark was silently discarded. If the first 30s of a
        video contains music or a non-speech intro, Whisper returns "" or ".",
        which collapses to 0 tokens after preprocessing and causes the text engine
        to return a neutral 0.5/0.5 result with no inference.

        Fix: chunk the raw audio into overlapping 25-second windows (5s overlap),
        transcribe each chunk independently, and join the results. This covers
        the full audio duration regardless of video length.

        Returns empty string on any failure so TextAnalyzer degrades gracefully.
        """
        if self._asr_processor is None or self._asr_model is None:
            return ""
        try:
            audio, _ = librosa.load(audio_wav, sr=SAMPLE_RATE, mono=True)
            audio = audio.astype("float32")

            # Whisper's hard context window is 30s (480 000 samples @ 16 kHz).
            # Use 25s chunks with 5s overlap so speech at chunk boundaries isn't cut.
            CHUNK_SAMPLES   = SAMPLE_RATE * 25   # 400 000 samples
            OVERLAP_SAMPLES = SAMPLE_RATE * 5    #  80 000 samples
            STEP_SAMPLES    = CHUNK_SAMPLES - OVERLAP_SAMPLES  # 320 000 samples

            total_samples = len(audio)
            chunks = []
            start = 0
            while start < total_samples:
                end = min(start + CHUNK_SAMPLES, total_samples)
                chunks.append(audio[start:end])
                if end == total_samples:
                    break
                start += STEP_SAMPLES

            parts = []
            for chunk in chunks:
                inputs   = self._asr_processor(
                    chunk, sampling_rate=SAMPLE_RATE, return_tensors="pt"
                )
                features = inputs.input_features.to(DEVICE)
                with torch.no_grad():
                    ids = self._asr_model.generate(
                        features,
                        language='ar',
                        task='transcribe',
                    )
                part = self._asr_processor.batch_decode(
                    ids, skip_special_tokens=True
                )[0].strip()
                if part:
                    parts.append(part)

            transcript = " ".join(parts).strip()
            print(f'[Pipeline] Transcript ({len(chunks)} chunks, {len(transcript)} chars): {transcript[:80]}')
            return transcript
        except Exception as e:
            print(f"[Pipeline] Transcription failed: {e}")
            return ""

    # ── Audio extraction ───────────────────────────────────────────────────────
    def _extract_audio(self, video_path: str, output_wav: str) -> bool:
        """Wraps extract_audio; returns True on success, False on any failure."""
        try:
            return extract_audio(video_path, output_wav)
        except Exception:
            return False

    # ── Video processing wrapper ──────────────────────────────────────────────
    @staticmethod
    def _video_error_result(reason: str) -> Dict[str, Any]:
        """
        [R2] FIX: VideoAnalyzer.process_video() has no internal try/except and
        can raise, unlike AudioAnalyzer.analyze()/TextAnalyzer.analyze() which
        always return a dict with label='Error' on failure. This builds a
        matching structured result so all three modalities fail the same way.
        """
        return {
            "score": 0.5,
            "label": "Error",
            "confidence": 0.0,
            "faces_detected": 0,
            "verdict_reason": reason,
            "peak_frame_time": 0.0,
            "all_probs": [],
            "temporal": {"variance": 0.0, "spikes": [], "temporal_score": 0.0},
            "roc_fig": None,
            "cm_fig": None,
            "video_path": None,
        }

    def _process_video(self, video_path: str) -> Dict[str, Any]:
        """Wraps VideoAnalyzer.process_video; never raises."""
        if self.video_analyzer is None:
            return self._video_error_result("Video model unavailable (init failed).")
        try:
            return self.video_analyzer.process_video(video_path)
        except Exception as e:
            return self._video_error_result(f"Video execution runtime failure: {e}")

    # ── Audio transcript filtering ────────────────────────────────────────────
    _AUDIO_PLACEHOLDER_MARKERS = (
        "No audio channel track identified",
        "Audio model unavailable",
        "Audio execution runtime failure",
    )

    @classmethod
    def _real_transcript(cls, audio_result: Dict[str, Any]) -> str:
        """
        [R1] FIX: AudioAnalyzer.analyze() returns non-empty placeholder
        sentences (no audio track / model unavailable / runtime failure)
        instead of an empty string. Those placeholders are long enough to
        skip TextAnalyzer.analyze()'s own empty-input guard, so the text
        engine was running real classification on placeholder text and
        returning misleading verdicts. Filter them out here before they
        ever reach TextAnalyzer.
        """
        transcript = audio_result.get("transcript", "") or ""
        if any(marker in transcript for marker in cls._AUDIO_PLACEHOLDER_MARKERS):
            return ""
        return transcript

    # ── Temp directory cleanup ─────────────────────────────────────────────────
    @staticmethod
    def _cleanup(tmp_dir: str, audio_wav: str) -> None:
        """
        [P1] FIX: clean up tmp_dir unconditionally (was only cleaned when audio_ok=True).
        Silently ignores OSError so cleanup never crashes the caller.
        """
        try:
            if os.path.exists(audio_wav):
                os.remove(audio_wav)
            if os.path.isdir(tmp_dir):
                os.rmdir(tmp_dir)
        except OSError:
            pass

    # ── Main pipeline ──────────────────────────────────────────────────────────
    def analyze_pipeline(self, video_path: str) -> Dict[str, Any]:
        results: Dict[str, Any] = {
            "video": {},
            "audio": {},
            "text":  {},
            "error": None,
        }

        # [Q4] FIX: validate video_path before doing any work
        if not video_path or not os.path.isfile(video_path):
            results["error"] = f"video_path does not exist or is not a file: {video_path!r}"
            return results

        # ── Stage 1: extract audio ─────────────────────────────────────────────
        tmp_dir   = tempfile.mkdtemp()
        audio_wav = os.path.join(tmp_dir, "extracted_audio.wav")
        audio_ok  = self._extract_audio(video_path, audio_wav)

        try:
            # ── Stages 2 & 3: maximum parallelism ─────────────────────────────
            #
            # Dependency graph:
            #   video  ──────────────────────────────────────────────────► video_result
            #   audio  ──► (transcript ready) ──► text ──────────────────► text_result
            #
            # "text" has a *data* dependency on "audio" (needs the transcript),
            # so it cannot start until audio finishes.  But it has NO dependency
            # on "video", so we must NOT block text on video.
            #
            # Strategy: run video + audio in parallel (3-worker pool).
            # The moment audio_future resolves, submit text as a third future —
            # it then overlaps with whatever time video still needs.
            #
            # [P2] FIX: outer try/except so any engine crash populates
            #           results['error'] instead of propagating raw.
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:

                    # Submit video immediately
                    video_future = pool.submit(self._process_video, video_path)

                    # [P3] FIX: short-circuit audio + transcription when extraction failed
                    if audio_ok:
                        audio_future      = pool.submit(self.audio_analyzer.analyze, audio_wav)
                        transcript_future = pool.submit(self._transcribe_audio, audio_wav)
                    else:
                        audio_future      = None
                        transcript_future = None

                    # Collect audio + transcript (both ran in parallel on audio_wav).
                    # IMPORTANT: both .result() calls must complete before we touch
                    # audio_wav again — _transcribe_audio holds an open librosa read
                    # on the file until its future resolves.
                    audio_result = audio_future.result()      if audio_future      is not None else {}
                    transcript   = transcript_future.result() if transcript_future is not None else ""

                    # transcript comes from _transcribe_audio (Whisper on full audio).
                    # audio_engine.analyze() no longer transcribes — its 'transcript' key is always ''.
                    # _real_transcript() is kept only to catch any residual placeholder strings.
                    transcript = transcript or self._real_transcript(
                        audio_result if isinstance(audio_result, dict) else {}
                    )
                    text_future = pool.submit(self.text_analyzer.analyze, transcript)

                    # Collect remaining futures (overlap with text engine)
                    video_result = video_future.result()
                    text_result  = text_future.result()

                    # [S1] FIX: delete audio_wav HERE — inside the with block, after
                    # ALL futures (including transcript_future) have resolved and
                    # returned their .result(). The outer finally below is kept only
                    # as a safety net for the error-exit path; if we reach this line
                    # the file is already gone and _cleanup() will no-op silently.
                    self._cleanup(tmp_dir, audio_wav)

            except Exception as exc:
                results["error"] = f"Engine error during parallel analysis: {exc}"
                return results

            results["video"] = video_result if isinstance(video_result, dict) else {}
            results["audio"] = audio_result if isinstance(audio_result, dict) else {}
            results["text"]  = text_result  if isinstance(text_result,  dict) else {}

        finally:
            # [P1] FIX: always clean up tmp_dir, regardless of audio_ok or errors.
            # [S1] On the happy path, audio_wav is already deleted above — this is
            #      a no-op safety net for the exception-exit path only.
            self._cleanup(tmp_dir, audio_wav)

        return results
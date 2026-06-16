"""
app.py
Multimodal Forensic Suite - Fixed Engine Initialization & Path Routing
"""

import torch  # Crucial: Must be imported first to resolve global type-hint namespace errors
import os
import sys
import json
import tempfile
import concurrent.futures
import numpy as np
import streamlit as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Setup absolute project path alignments
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from inference_engines.video_engine import VideoAnalyzer
from inference_engines.audio_engine import AudioAnalyzer, extract_audio
from inference_engines.text_engine  import TextAnalyzer

import warnings
warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Deepfake Forensic Suite",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ── HTML render helper ────────────────────────────────────────────────────────
def render_html(content: str) -> None:
    """
    Render raw HTML/CSS through st.markdown safely.

    Streamlit's Markdown parser treats any line beginning with 4+ spaces as a
    literal code block. Because our HTML f-strings are written inside nested
    `with` / `if` blocks, every line carries that indentation, which causes
    the <div> markup to be printed as plain text instead of rendered.
    Stripping leading whitespace from every line fixes this without changing
    the rendered output (browsers collapse leading whitespace anyway).
    """
    st.markdown(
        "\n".join(line.lstrip() for line in content.split("\n")),
        unsafe_allow_html=True
    )


# ── Classic Blue Design System ────────────────────────────────────────────────
# Palette:
#   Primary   #0D3B66  Classic blue     — headings, badges, borders
#   Accent    #7F9DB1  Dusty blue       — labels, subtext, secondary UI
#   BG        #B4D6E3  Soft sky blue    — page background
#   Panel BG  #dceef5  lighter tint     — card surfaces
#   Real      #16a34a  Green            — real verdict bars / chips
#   Fake      #dc2626  Red              — fake verdict bars / chips
render_html("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, .stApp {
    background: #B4D6E3 !important;
    color: #0D3B66 !important;
    font-family: 'Inter', sans-serif !important;
}
.stButton button, .stCheckbox label, .stFileUploader label,
label[data-testid], .stSpinner p { color: #0D3B66 !important; }
.stButton button {
    background: #0D3B66 !important; color: #ffffff !important;
    border: none !important; border-radius: 6px !important;
}
.stButton button:hover { background: #0a2d50 !important; }

.sv-header {
    margin-bottom: 24px; border-bottom: 2px solid #7F9DB1; padding-bottom: 14px;
}
.sv-title    { font-size: 24px; font-weight: 700; color: #0D3B66; letter-spacing: -0.5px; }
.sv-subtitle { font-size: 13px; color: #7F9DB1; margin-top: 4px; }

.sv-panel {
    background: #dceef5; border: 1px solid #7F9DB1; border-radius: 8px;
    padding: 20px; margin-bottom: 20px;
}
.sv-panel-header {
    display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;
}
.sv-panel-name  { font-size: 15px; font-weight: 600; color: #0D3B66; }
.sv-panel-model { font-size: 11px; color: #7F9DB1; font-family: 'JetBrains Mono', monospace; }

.sv-badge { padding: 4px 10px; border-radius: 4px; font-size: 11px; font-weight: 600; letter-spacing: 0.5px; }
.sv-badge-real { background: rgba(22,163,74,.15);  color: #15803d; border: 1px solid rgba(22,163,74,.4); }
.sv-badge-fake { background: rgba(220,38,38,.15);  color: #dc2626; border: 1px solid rgba(220,38,38,.4); }

.sv-metric-row  { display: flex; gap: 16px; margin-bottom: 14px; background: #c8e4ed; padding: 12px; border-radius: 6px; }
.sv-metric-card { flex: 1; }
.sv-metric-label { font-size: 10px; color: #7F9DB1; text-transform: uppercase; letter-spacing: 0.5px; }
.sv-metric-value { font-size: 18px; font-weight: 600; color: #0D3B66; margin-top: 2px; }
.sv-metric-sub   { font-size: 10px; color: #7F9DB1; margin-top: 2px; font-family: 'JetBrains Mono', monospace; }

.sv-progress-container { margin-top: 12px; }
.sv-progress-label { display: flex; justify-content: space-between; font-size: 11px; color: #7F9DB1; margin-bottom: 4px; }
.sv-progress-bar-bg   { background: #a8cdd9; height: 6px; border-radius: 3px; overflow: hidden; }
.sv-progress-bar-fill { height: 100%; border-radius: 3px; transition: width .6s ease; }
.sv-fill-real { background: #16a34a; }
.sv-fill-fake { background: #dc2626; }

.sv-prob-row { display: flex; gap: 10px; margin-top: 12px; }
.sv-prob-chip { flex: 1; padding: 8px 10px; border-radius: 6px; border: 1px solid; text-align: center; }
.sv-prob-chip-real { background: rgba(22,163,74,.10); border-color: rgba(22,163,74,.35); }
.sv-prob-chip-fake { background: rgba(220,38,38,.10); border-color: rgba(220,38,38,.35); }
.sv-prob-chip-label { font-size: 10px; color: #7F9DB1; text-transform: uppercase; letter-spacing: .5px; }
.sv-prob-chip-val   { font-size: 18px; font-weight: 700; margin-top: 2px; font-family: 'JetBrains Mono', monospace; }
.sv-prob-chip-real .sv-prob-chip-val { color: #16a34a; }
.sv-prob-chip-fake .sv-prob-chip-val { color: #dc2626; }

.sv-diag-row { display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid #a8cdd9; font-size: 12px; }
.sv-diag-row:last-child { border-bottom: none; }
.sv-diag-key      { color: #7F9DB1; }
.sv-diag-val      { color: #0D3B66; font-family: 'JetBrains Mono', monospace; font-weight: 500; }
.sv-diag-val-warn { color: #dc2626; font-family: 'JetBrains Mono', monospace; font-weight: 500; }

.sv-token-wrap { line-height: 2.4; font-size: 13px; }
.sv-token { display: inline-block; padding: 2px 5px; border-radius: 3px; margin: 2px 1px; cursor: default; }

.sv-transcript-box {
    background: #c8e4ed; border: 1px solid #7F9DB1; border-radius: 6px; padding: 14px;
    font-size: 13px; line-height: 1.6; color: #0D3B66;
    max-height: 180px; overflow-y: auto; direction: rtl;
}
.sv-section-label {
    font-size: 10px; letter-spacing: 1.2px; text-transform: uppercase;
    color: #7F9DB1; margin-bottom: 8px; margin-top: 14px;
}
.sv-placeholder {
    background: #c8e4ed; border: 1px solid #7F9DB1; border-radius: 6px;
    padding: 16px; text-align: center; color: #7F9DB1; font-size: 12px; margin-bottom: 12px;
}
</style>
""")

# ── Application Header ────────────────────────────────────────────────────────
render_html("""
<div class="sv-header">
    <div class="sv-title">Deepfake Forensic Suite</div>
    <div class="sv-subtitle">Multimodal Manipulation &amp; Arabic Fake News Verification Engine</div>
</div>
""")


# ── Helper: score bar ─────────────────────────────────────────────────────────
def _bar(label: str, pct: float, fill_cls: str, explain: str = "") -> str:
    ex = f"<div style='font-size:10px;color:#7F9DB1;margin-top:3px;'>{explain}</div>" if explain else ""
    return f"""
    <div class="sv-progress-container">
        <div class="sv-progress-label"><span>{label}</span><span>{pct:.1f}%</span></div>
        <div class="sv-progress-bar-bg">
            <div class="sv-progress-bar-fill {fill_cls}" style="width:{min(pct,100):.1f}%;"></div>
        </div>{ex}
    </div>"""


# ── Helper: token importance highlight ───────────────────────────────────────
def _token_highlights(tokens: list, scores: list) -> str:
    if not tokens:
        return "<span style='color:#7F9DB1;font-size:12px;'>No tokens available.</span>"

    arr  = np.array(scores if scores else [0.0] * len(tokens), dtype=float)
    mn, mx = arr.min(), arr.max()
    denom = (mx - mn + 1e-8)
    norm = (arr - mn) / denom

    html = "<div class='sv-token-wrap' dir='rtl'>"
    for idx, (tok, w) in enumerate(zip(tokens, norm)):
        # Gradient: low=dusty-blue #7F9DB1 → high=red #dc2626
        r   = int(0x7f + (0xdc - 0x7f) * w)
        g   = int(0x9d + (0x26 - 0x9d) * w)
        b   = int(0xb1 + (0x26 - 0xb1) * w)
        fg  = f"rgb({r},{g},{b})"
        bg  = f"rgba({r},{g},{b},{0.12 + 0.45 * w:.2f})"
        brd = f"rgba({r},{g},{b},{0.2 + 0.4 * w:.2f})"
        raw_score = float(arr[idx])
        html += (
            f"<span class='sv-token' "
            f"style='color:{fg};background:{bg};border:1px solid {brd};' "
            f"title='token: {tok}  |  importance: {raw_score:.4f}'>{tok}</span>"
        )
    html += "</div>"
    return html


# ── Helper: load video model metadata ────────────────────────────────────────
def _video_meta(weights_dir: str) -> dict:
    path = os.path.join(weights_dir, "model_metadata.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {
        "best_auc": 0.9916, "best_threshold": 0.8504,
        "architecture": "EfficientNet-B0 + GeM",
        "dataset": "fakereal-dataset",
        "epochs_trained": 15, "training_frames": 137990,
    }


# ── Instantiate engines (cached) ─────────────────────────────────────────────
@st.cache_resource
def init_analyzers():
    # Setup path configurations to look inside 'model_weights' root folder
    base_weights = os.path.join(current_dir, "model_weights")

    os.makedirs(os.path.join(base_weights, "deepfake_vision"), exist_ok=True)
    os.makedirs(os.path.join(base_weights, "whisper_audio"), exist_ok=True)
    os.makedirs(os.path.join(base_weights, "arabert_news"), exist_ok=True)

    # Locate or establish absolute file pathing for the weight binary
    audio_ckpt = os.path.join(base_weights, "whisper_audio", "checkpoint.pth")
    if not os.path.exists(audio_ckpt):
        audio_ckpt = "aubmindlab/bert-base-arabertv02"  # Online fallback signature identifier

    # Core engine parameter binding matching backends exactly
    v = VideoAnalyzer(model_path=os.path.join(base_weights, "deepfake_vision", "best_model.pth"))
    a = AudioAnalyzer(checkpoint_path=audio_ckpt)
    t = TextAnalyzer(model_dir=os.path.join(base_weights, "arabert_news"))
    return v, a, t

v_eng, a_eng, t_eng = init_analyzers()

# ── Split-screen layout ───────────────────────────────────────────────────────
left_col, right_col = st.columns([1, 2], gap="large")

with left_col:
    st.markdown(
        '<p style="font-size:14px;font-weight:600;margin-bottom:8px;color:#0D3B66;">Source Medium Ingestion</p>',
        unsafe_allow_html=True
    )
    uploaded_file = st.file_uploader(
        "Upload video file (.mp4, .avi, .mov)",
        type=["mp4", "avi", "mov"],
        label_visibility="collapsed"
    )
    news_mode       = st.checkbox("News Broadcast Mode Tuning", value=False)
    v_eng.news_mode = news_mode
    v_eng.threshold = 0.67 if news_mode else 0.72

    if uploaded_file is not None:
        st.markdown('<div style="margin-top:14px;"></div>', unsafe_allow_html=True)
        st.video(uploaded_file)
        st.markdown('<div style="margin-top:14px;"></div>', unsafe_allow_html=True)
        analyze_btn = st.button("Execute Full Diagnostic Analysis", use_container_width=True)
    else:
        analyze_btn = False

# ── Right column ──────────────────────────────────────────────────────────────
with right_col:

    if not analyze_btn:
        render_html("""
        <div style="border:2px dashed #7F9DB1;border-radius:8px;padding:60px 20px;
                    text-align:center;color:#0D3B66;">
            <div style="font-size:14px;">Awaiting file selection input parameters...</div>
            <div style="font-size:11px;margin-top:4px;">
                Upload a source media channel stream on the left pane and select execute
                to trigger multimodal checks.
            </div>
        </div>""")

    if analyze_btn and uploaded_file is not None:

        with tempfile.TemporaryDirectory() as tmpdir:
            temp_video_path = os.path.join(tmpdir, "input_media.mp4")
            temp_audio_path = os.path.join(tmpdir, "extracted_audio.wav")

            with open(temp_video_path, "wb") as fh:
                fh.write(uploaded_file.read())

            with st.spinner("Extracting audio channels and allocating thread vectors..."):
                audio_extracted = extract_audio(temp_video_path, temp_audio_path)                

            with st.spinner("Processing simultaneous model layer diagnostics..."):
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                    vf = pool.submit(v_eng.process_video, temp_video_path)
                    af = pool.submit(a_eng.analyze, temp_audio_path if audio_extracted else "")
                    video_res = vf.result()
                    audio_res = af.result()

                if not isinstance(video_res, dict): video_res = {}
                if not isinstance(audio_res, dict): audio_res = {}

                extracted_text = audio_res.get('transcript', '')
                text_res       = t_eng.analyze(extracted_text)
                if not isinstance(text_res, dict): text_res = {}

            # ── Unpack VIDEO ──────────────────────────────────────────────
            v_label     = video_res.get('label',           'Real')
            v_score     = float(video_res.get('score',     0.0))
            v_conf      = float(video_res.get('confidence',0.0))
            v_faces     = video_res.get('faces_detected',  0)
            v_reason    = video_res.get('verdict_reason',  '—')
            v_peak_t    = float(video_res.get('peak_frame_time', 0.0))
            v_temporal  = video_res.get('temporal',        {})
            v_spikes    = v_temporal.get('spikes',         [])
            v_variance  = float(v_temporal.get('variance', 0.0))
            v_consist   = float(v_temporal.get('temporal_score', 1.0))
            v_all_probs = video_res.get('all_probs', video_res.get('frame_scores', []))
            v_roc_fig   = video_res.get('roc_fig',  None)
            v_cm_fig    = video_res.get('cm_fig',   None)
            v_meta      = _video_meta(os.path.join(current_dir, "model_weights", "deepfake_vision"))
            v_best_auc  = float(v_meta.get('best_auc',       0.9916))
            v_best_thr  = float(v_meta.get('best_threshold', 0.8504))

            # ── Unpack AUDIO ──────────────────────────────────────────────
            a_label      = audio_res.get('label',        'Real')
            a_score      = float(audio_res.get('score',  0.0))
            a_conf       = float(audio_res.get('confidence', 0.0))
            a_prob_real  = float(audio_res.get('prob_real',  1.0 - a_score))
            a_prob_fake  = float(audio_res.get('prob_fake',  a_score))
            a_manip_segs = audio_res.get('manipulated_segments',  'None detected')
            a_peak_time  = audio_res.get('peak_suspicion_time',   None)
            a_susp_bands = audio_res.get('suspicious_freq_bands', None)
            a_result_plot= audio_res.get('result_plot',           None)

            # ── Unpack TEXT ───────────────────────────────────────────────
            t_label      = text_res.get('label',      'Real')
            t_conf       = float(text_res.get('confidence', 0.5))
            t_prob_real  = float(text_res.get('prob_real',
                                              1.0 - t_conf if t_label == 'Fake' else t_conf))
            t_prob_fake  = float(text_res.get('prob_fake',
                                              t_conf if t_label == 'Fake' else 1.0 - t_conf))
            t_tokens     = text_res.get('tokens',           [])
            t_importance = text_res.get('token_importance', [])

            # ── Derived flags & CSS classes ───────────────────────────────
            v_is_fake = v_label == 'Fake'
            a_is_fake = a_label == 'Fake'
            t_is_fake = t_label == 'Fake'

            v_badge_cls = "sv-badge-fake" if v_is_fake else "sv-badge-real"
            a_badge_cls = "sv-badge-fake" if a_is_fake else "sv-badge-real"
            t_badge_cls = "sv-badge-fake" if t_is_fake else "sv-badge-real"
            v_fill = "sv-fill-fake" if v_is_fake else "sv-fill-real"
            a_fill = "sv-fill-fake" if a_is_fake else "sv-fill-real"
            t_fill = "sv-fill-fake" if t_is_fake else "sv-fill-real"

            v_spike_cls  = "sv-diag-val-warn" if v_spikes    else "sv-diag-val"
            v_var_cls    = "sv-diag-val-warn" if v_variance > 0.05 else "sv-diag-val"
            a_peak_str   = f"{a_peak_time:.2f}s" if isinstance(a_peak_time, (int, float)) \
                           else (str(a_peak_time) if a_peak_time else "—")
            a_bands_str  = str(a_susp_bands) if a_susp_bands else "—"
            a_seg_cls    = "sv-diag-val-warn" if str(a_manip_segs) not in ("None detected","") \
                           else "sv-diag-val"

            # =============================================================
            # PANEL 1 — VIDEO FORENSIC ANALYSIS
            # =============================================================
            v_bar_html = _bar("Classification confidence", v_conf * 100, v_fill,
                              "Sigmoid output probability that this video contains deepfake manipulation.")

            render_html(f"""
            <div class="sv-panel">
                <div class="sv-panel-header">
                    <div>
                        <div class="sv-panel-name">Video Core Visual Analysis</div>
                        <div class="sv-panel-model">EfficientNet-B0 · Spatial Frame Ensemble · GeM Pooling · AUC {v_best_auc:.4f}</div>
                    </div>
                    <span class="sv-badge {v_badge_cls}">{v_label.upper()}</span>
                </div>

                <div class="sv-metric-row">
                    <div class="sv-metric-card">
                        <div class="sv-metric-label">Sigmoid Score</div>
                        <div class="sv-metric-value">{v_score:.4f}</div>
                        <div class="sv-metric-sub">all_probs mean</div>
                    </div>
                    <div class="sv-metric-card">
                        <div class="sv-metric-label">Face Frames</div>
                        <div class="sv-metric-value">{v_faces} / 16</div>
                        <div class="sv-metric-sub">MTCNN detected</div>
                    </div>
                    <div class="sv-metric-card">
                        <div class="sv-metric-label">Peak Anomaly</div>
                        <div class="sv-metric-value">{v_peak_t:.1f}s</div>
                        <div class="sv-metric-sub">highest-score frame</div>
                    </div>
                    <div class="sv-metric-card">
                        <div class="sv-metric-label">Temporal Consistency</div>
                        <div class="sv-metric-value">{v_consist:.2f}</div>
                        <div class="sv-metric-sub">1.0 = stable</div>
                    </div>
                </div>

                {v_bar_html}

                <div class="sv-section-label">Engine Diagnostics  (model_metadata.json)</div>
                <div style="background:#c8e4ed;border:1px solid #a8cdd9;border-radius:6px;padding:10px 14px;">
                    <div class="sv-diag-row">
                        <span class="sv-diag-key">Peak Validation AUC</span>
                        <span class="sv-diag-val">{v_best_auc:.4f}</span>
                    </div>
                    <div class="sv-diag-row">
                        <span class="sv-diag-key">Optimal PR-F1 Threshold</span>
                        <span class="sv-diag-val">{v_best_thr:.4f}</span>
                    </div>
                    <div class="sv-diag-row">
                        <span class="sv-diag-key">Architecture</span>
                        <span class="sv-diag-val">{v_meta.get('architecture','EfficientNet-B0 + GeM')}</span>
                    </div>
                    <div class="sv-diag-row">
                        <span class="sv-diag-key">Training frames</span>
                        <span class="sv-diag-val">{v_meta.get('training_frames',137990):,}</span>
                    </div>
                    <div class="sv-diag-row">
                        <span class="sv-diag-key">Frame-score variance</span>
                        <span class="{v_var_cls}">{v_variance:.6f}</span>
                    </div>
                    <div class="sv-diag-row">
                        <span class="sv-diag-key">Spike frames</span>
                        <span class="{v_spike_cls}">{str(v_spikes) if v_spikes else "none"}</span>
                    </div>
                    <div class="sv-diag-row">
                        <span class="sv-diag-key">Verdict trace</span>
                        <span class="sv-diag-val" style="font-size:10px;max-width:58%;text-align:right;
                              white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{v_reason}</span>
                    </div>
                </div>
            </div>
            """)

            # Frame-level all_probs bar chart
            if len(v_all_probs) > 0:
                render_html(
                    "<div class='sv-section-label' style='margin-top:0;'>"
                    "Frame-level sigmoid scores  (all_probs array)</div>"
                )
                fig_fp, ax = plt.subplots(figsize=(7, 1.7))
                fig_fp.patch.set_facecolor('#dceef5')
                ax.set_facecolor('#dceef5')
                xs     = list(range(len(v_all_probs)))
                cols   = ['#dc2626' if s > v_eng.threshold else '#16a34a' for s in v_all_probs]
                ax.bar(xs, v_all_probs, color=cols, width=0.7, zorder=3)
                ax.axhline(v_eng.threshold, color='#0D3B66', linestyle='--', lw=1,
                           label=f'threshold {v_eng.threshold:.2f}')
                ax.set_ylim(0, 1); ax.set_xlim(-0.5, len(v_all_probs) - 0.5)
                ax.tick_params(colors='#0D3B66', labelsize=8)
                ax.grid(axis='y', color='#a8cdd9', linewidth=0.5, zorder=0)
                for sp in ax.spines.values(): sp.set_edgecolor('#7F9DB1')
                ax.legend(fontsize=7, facecolor='#dceef5', edgecolor='#7F9DB1', labelcolor='#0D3B66')
                plt.tight_layout()
                st.pyplot(fig_fp, use_container_width=True)
                plt.close(fig_fp)

            with st.expander("📊  Analytics & Performance Report  (ROC · Confusion Matrix)", expanded=False):
                c1, c2 = st.columns(2)
                with c1:
                    if v_roc_fig is not None:
                        st.pyplot(v_roc_fig, use_container_width=True)
                    else:
                        render_html("""
                        <div class="sv-placeholder">
                            ROC curve not returned by engine.
                        </div>""")
                with c2:
                    if v_cm_fig is not None:
                        st.pyplot(v_cm_fig, use_container_width=True)
                    else:
                        render_html("""
                        <div class="sv-placeholder">
                            Confusion matrix not returned by engine.
                        </div>""")

            # =============================================================
            # PANEL 2 — AUDIO FORENSIC ANALYSIS
            # =============================================================
            a_bar_html = _bar("Acoustic authenticity confidence", a_conf * 100, a_fill,
                              "Whisper encoder classification certainty for this audio track.")

            render_html(f"""
            <div class="sv-panel">
                <div class="sv-panel-header">
                    <div>
                        <div class="sv-panel-name">Audio Frequency Track Analysis</div>
                        <div class="sv-panel-model">Whisper-small Encoder · 8-Dialect ArFake · Gradient Saliency</div>
                    </div>
                    <span class="sv-badge {a_badge_cls}">{a_label.upper()}</span>
                </div>

                <div class="sv-metric-row">
                    <div class="sv-metric-card">
                        <div class="sv-metric-label">Raw Score</div>
                        <div class="sv-metric-value">{a_score:.4f}</div>
                        <div class="sv-metric-sub">softmax fake logit</div>
                    </div>
                    <div class="sv-metric-card">
                        <div class="sv-metric-label">Confidence</div>
                        <div class="sv-metric-value">{a_conf*100:.1f}%</div>
                        <div class="sv-metric-sub">verdict certainty</div>
                    </div>
                    <div class="sv-metric-card">
                        <div class="sv-metric-label">Anomaly Segments</div>
                        <div class="sv-metric-value" style="font-size:14px;">{a_manip_segs}</div>
                        <div class="sv-metric-sub">suspicious windows</div>
                    </div>
                </div>

                <div class="sv-prob-row">
                    <div class="sv-prob-chip sv-prob-chip-real">
                        <div class="sv-prob-chip-label">P(Real)</div>
                        <div class="sv-prob-chip-val">{a_prob_real:.4f}</div>
                    </div>
                    <div class="sv-prob-chip sv-prob-chip-fake">
                        <div class="sv-prob-chip-label">P(Fake)</div>
                        <div class="sv-prob-chip-val">{a_prob_fake:.4f}</div>
                    </div>
                </div>

                {a_bar_html}

                <div class="sv-section-label">Computed Diagnostic Triggers</div>
                <div style="background:#c8e4ed;border:1px solid #a8cdd9;border-radius:6px;padding:10px 14px;">
                    <div class="sv-diag-row">
                        <span class="sv-diag-key">Peak suspicion time</span>
                        <span class="{"sv-diag-val-warn" if a_is_fake else "sv-diag-val"}">{a_peak_str}</span>
                    </div>
                    <div class="sv-diag-row">
                        <span class="sv-diag-key">Suspicious frequency bands</span>
                        <span class="{"sv-diag-val-warn" if a_susp_bands else "sv-diag-val"}">{a_bands_str}</span>
                    </div>
                    <div class="sv-diag-row">
                        <span class="sv-diag-key">Manipulated segments</span>
                        <span class="{a_seg_cls}">{a_manip_segs}</span>
                    </div>
                </div>
            </div>
            """)

            if a_result_plot is not None:
                render_html("<div class='sv-section-label'>Waveform · Saliency Curve · Spectrogram</div>")
                st.pyplot(a_result_plot, use_container_width=True)
            else:
                render_html("""
                <div class="sv-placeholder">
                    Waveform / saliency dashboard not returned by audio engine.
                </div>""")

            # =============================================================
            # PANEL 3 — TEXT SEMANTIC ANALYSIS
            # =============================================================
            t_bar_html = _bar("Information authenticity prediction", t_conf * 100, t_fill,
                              "AraBERT classification confidence across 606,912 AFND article embeddings.")

            token_html = _token_highlights(t_tokens, t_importance)

            render_html(f"""
            <div class="sv-panel">
                <div class="sv-panel-header">
                    <div>
                        <div class="sv-panel-name">Arabic Text Semantic Analysis</div>
                        <div class="sv-panel-model">AraBERT v02 · AFND 606k Articles · Gradient × Embedding Norm</div>
                    </div>
                    <span class="sv-badge {t_badge_cls}">{t_label.upper()}</span>
                </div>

                <div class="sv-prob-row">
                    <div class="sv-prob-chip sv-prob-chip-real">
                        <div class="sv-prob-chip-label">P(Real)</div>
                        <div class="sv-prob-chip-val">{t_prob_real:.4f}</div>
                    </div>
                    <div class="sv-prob-chip sv-prob-chip-fake">
                        <div class="sv-prob-chip-label">P(Fake)</div>
                        <div class="sv-prob-chip-val">{t_prob_fake:.4f}</div>
                    </div>
                </div>

                {t_bar_html}

                <div class="sv-section-label">Token Importance Map  (gradient × embedding norm)</div>
                <div style="background:#c8e4ed;border:1px solid #7F9DB1;border-radius:6px;
                            padding:12px 14px;max-height:200px;overflow-y:auto;">
                    {token_html}
                </div>
                <div style="margin-top:6px;font-size:10px;color:#7F9DB1;line-height:1.5;">
                    Colour intensity = influence on classification.
                    <span style="color:#dc2626;">■</span> High &nbsp;·&nbsp;
                    <span style="color:#a8cdd9;border:1px solid #7F9DB1;padding:0 3px;">■</span> Low
                </div>

                <div class="sv-section-label">Extracted Speech Transcript (Arabic Language Context)</div>
            </div>
            """)

            render_html(
                f'<div class="sv-transcript-box">{extracted_text or "No transcript extracted."}</div>'
            )
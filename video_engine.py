# """
# app.py
# Multimodal Forensic Suite - Production User Dashboard UI Layout
# Updated: Full backend output injection for all three inference pipelines.
# """

# import os
# import sys
# import json
# import tempfile
# import concurrent.futures
# import numpy as np
# import streamlit as st
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt

# # Setup absolute project path alignments
# current_dir = os.path.dirname(os.path.abspath(__file__))
# if current_dir not in sys.path:
#     sys.path.append(current_dir)

# from inference_engines.video_engine import VideoAnalyzer
# from inference_engines.audio_engine import AudioAnalyzer, extract_audio
# from inference_engines.text_engine  import TextAnalyzer

# # ── Page config ───────────────────────────────────────────────────────────────
# st.set_page_config(
#     page_title="Deepfake Forensic Suite",
#     layout="wide",
#     initial_sidebar_state="collapsed"
# )

# # ── Sapphire Veil Design System — unchanged from original ────────────────────
# st.markdown("""
# <style>
# @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

# html, body, .stApp {
#     background: #0a1628 !important;
#     color: #e8f0fa !important;
#     font-family: 'Inter', sans-serif !important;
# }
# .sv-header {
#     margin-bottom: 24px; border-bottom: 1px solid #1a3050; padding-bottom: 14px;
# }
# .sv-title  { font-size: 24px; font-weight: 700; color: #ffffff; letter-spacing: -0.5px; }
# .sv-subtitle { font-size: 13px; color: #5a8ab8; margin-top: 4px; }

# .sv-panel {
#     background: #0d1e35; border: 1px solid #1a3050; border-radius: 8px;
#     padding: 20px; margin-bottom: 20px;
# }
# .sv-panel-header {
#     display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;
# }
# .sv-panel-name  { font-size: 15px; font-weight: 600; color: #ffffff; }
# .sv-panel-model { font-size: 11px; color: #4b7bb0; font-family: 'JetBrains Mono', monospace; }

# .sv-badge { padding: 4px 10px; border-radius: 4px; font-size: 11px; font-weight: 600; letter-spacing: 0.5px; }
# .sv-badge-real { background: rgba(16,185,129,.15); color: #10b981; border: 1px solid rgba(16,185,129,.3); }
# .sv-badge-fake { background: rgba(239,68,68,.15);  color: #ef4444; border: 1px solid rgba(239,68,68,.3); }

# .sv-metric-row  { display: flex; gap: 16px; margin-bottom: 14px; background: #091525; padding: 12px; border-radius: 6px; }
# .sv-metric-card { flex: 1; }
# .sv-metric-label { font-size: 10px; color: #5a8ab8; text-transform: uppercase; letter-spacing: 0.5px; }
# .sv-metric-value { font-size: 18px; font-weight: 600; color: #ffffff; margin-top: 2px; }
# .sv-metric-sub   { font-size: 10px; color: #4b7bb0; margin-top: 2px; font-family: 'JetBrains Mono', monospace; }

# .sv-progress-container { margin-top: 12px; }
# .sv-progress-label { display: flex; justify-content: space-between; font-size: 11px; color: #5a8ab8; margin-bottom: 4px; }
# .sv-progress-bar-bg   { background: #162a45; height: 6px; border-radius: 3px; overflow: hidden; }
# .sv-progress-bar-fill { height: 100%; border-radius: 3px; transition: width .6s ease; }
# .sv-fill-real { background: #10b981; }
# .sv-fill-fake { background: #ef4444; }

# .sv-prob-row { display: flex; gap: 10px; margin-top: 12px; }
# .sv-prob-chip { flex: 1; padding: 8px 10px; border-radius: 6px; border: 1px solid; text-align: center; }
# .sv-prob-chip-real { background: rgba(16,185,129,.08); border-color: rgba(16,185,129,.25); }
# .sv-prob-chip-fake { background: rgba(239,68,68,.08);  border-color: rgba(239,68,68,.25); }
# .sv-prob-chip-label { font-size: 10px; color: #5a8ab8; text-transform: uppercase; letter-spacing: .5px; }
# .sv-prob-chip-val   { font-size: 18px; font-weight: 700; margin-top: 2px; font-family: 'JetBrains Mono', monospace; }
# .sv-prob-chip-real .sv-prob-chip-val { color: #10b981; }
# .sv-prob-chip-fake .sv-prob-chip-val { color: #ef4444; }

# .sv-diag-row { display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid #162a45; font-size: 12px; }
# .sv-diag-row:last-child { border-bottom: none; }
# .sv-diag-key      { color: #5a8ab8; }
# .sv-diag-val      { color: #e8f0fa; font-family: 'JetBrains Mono', monospace; font-weight: 500; }
# .sv-diag-val-warn { color: #ef4444; font-family: 'JetBrains Mono', monospace; font-weight: 500; }

# .sv-token-wrap { line-height: 2.4; font-size: 13px; }
# .sv-token { display: inline-block; padding: 2px 5px; border-radius: 3px; margin: 2px 1px; cursor: default; }

# .sv-transcript-box {
#     background: #091525; border: 1px solid #1a3050; border-radius: 6px; padding: 14px;
#     font-size: 13px; line-height: 1.6; color: #b8cee6;
#     max-height: 180px; overflow-y: auto; direction: rtl;
# }
# .sv-section-label {
#     font-size: 10px; letter-spacing: 1.2px; text-transform: uppercase;
#     color: #3d6a99; margin-bottom: 8px; margin-top: 14px;
# }
# .sv-placeholder {
#     background: #091525; border: 1px solid #1a3050; border-radius: 6px;
#     padding: 16px; text-align: center; color: #3d6a99; font-size: 12px; margin-bottom: 12px;
# }
# </style>
# """, unsafe_allow_html=True)

# # ── Application Header ────────────────────────────────────────────────────────
# st.markdown("""
# <div class="sv-header">
#     <div class="sv-title">Deepfake Forensic Suite</div>
#     <div class="sv-subtitle">Multimodal Manipulation &amp; Arabic Fake News Verification Engine</div>
# </div>
# """, unsafe_allow_html=True)


# # ── Helper: score bar ─────────────────────────────────────────────────────────
# def _bar(label: str, pct: float, fill_cls: str, explain: str = "") -> str:
#     ex = f"<div style='font-size:10px;color:#3d6a99;margin-top:3px;'>{explain}</div>" if explain else ""
#     return f"""
#     <div class="sv-progress-container">
#         <div class="sv-progress-label"><span>{label}</span><span>{pct:.1f}%</span></div>
#         <div class="sv-progress-bar-bg">
#             <div class="sv-progress-bar-fill {fill_cls}" style="width:{min(pct,100):.1f}%;"></div>
#         </div>{ex}
#     </div>"""


# # ── Helper: token importance highlight ───────────────────────────────────────
# def _token_highlights(tokens: list, scores: list) -> str:
#     """
#     Colour each token by its gradient × embedding norm importance weight.
#     Low importance  → dark slate  (#1a3050)
#     High importance → alert red   (#ef4444)
#     """
#     if not tokens:
#         return "<span style='color:#5a8ab8;font-size:12px;'>No tokens available.</span>"

#     arr  = np.array(scores if scores else [0.0] * len(tokens), dtype=float)
#     mn, mx = arr.min(), arr.max()
#     norm = (arr - mn) / (mx - mn + 1e-8)

#     html = "<div class='sv-token-wrap' dir='rtl'>"
#     for tok, w in zip(tokens, norm):
#         r   = int(0x1a + (0xef - 0x1a) * w)
#         g   = int(0x30 + (0x44 - 0x30) * w)
#         b   = int(0x50 + (0x44 - 0x50) * w)
#         fg  = f"rgb({r},{g},{b})"
#         bg  = f"rgba({r},{g},{b},{0.12 + 0.45 * w:.2f})"
#         brd = f"rgba({r},{g},{b},{0.2 + 0.4 * w:.2f})"
#         raw_score = float(arr[list(tokens).index(tok)]) if tok in tokens else 0.0
#         html += (
#             f"<span class='sv-token' "
#             f"style='color:{fg};background:{bg};border:1px solid {brd};' "
#             f"title='token: {tok}  |  importance: {raw_score:.4f}'>{tok}</span>"
#         )
#     html += "</div>"
#     return html


# # ── Helper: load video model metadata ────────────────────────────────────────
# def _video_meta(weights_dir: str) -> dict:
#     path = os.path.join(weights_dir, "model_metadata.json")
#     if os.path.exists(path):
#         with open(path) as f:
#             return json.load(f)
#     return {
#         "best_auc": 0.9916, "best_threshold": 0.8504,
#         "architecture": "EfficientNet-B0 + GeM",
#         "dataset": "fakereal-dataset",
#         "epochs_trained": 15, "training_frames": 137990,
#     }


# # ── Instantiate engines (cached) ─────────────────────────────────────────────
# @st.cache_resource
# def init_analyzers():
#     v = VideoAnalyzer(checkpoint_path="model_weights/deepfake_vision/best_model.pth")
#     a = AudioAnalyzer(checkpoint_path="model_weights/whisper_audio")
#     t = TextAnalyzer(model_dir="model_weights/arabert_news")
#     return v, a, t

# v_eng, a_eng, t_eng = init_analyzers()

# # ── Split-screen layout ───────────────────────────────────────────────────────
# left_col, right_col = st.columns([1, 2], gap="large")

# with left_col:
#     st.markdown(
#         '<p style="font-size:14px;font-weight:600;margin-bottom:8px;">Source Medium Ingestion</p>',
#         unsafe_allow_html=True
#     )
#     uploaded_file = st.file_uploader(
#         "Upload video file (.mp4, .avi, .mov)",
#         type=["mp4", "avi", "mov"],
#         label_visibility="collapsed"
#     )
#     news_mode       = st.checkbox("News Broadcast Mode Tuning", value=False)
#     v_eng.news_mode = news_mode
#     v_eng.threshold = 0.67 if news_mode else 0.72

#     if uploaded_file is not None:
#         st.markdown('<div style="margin-top:14px;"></div>', unsafe_allow_html=True)
#         st.video(uploaded_file)
#         st.markdown('<div style="margin-top:14px;"></div>', unsafe_allow_html=True)
#         analyze_btn = st.button("Execute Full Diagnostic Analysis", use_container_width=True)
#     else:
#         analyze_btn = False

# # ── Right column ──────────────────────────────────────────────────────────────
# with right_col:

#     if not analyze_btn:
#         st.markdown("""
#         <div style="border:2px dashed #1a3050;border-radius:8px;padding:60px 20px;
#                     text-align:center;color:#5a8ab8;">
#             <div style="font-size:14px;">Awaiting file selection input parameters...</div>
#             <div style="font-size:11px;margin-top:4px;">
#                 Upload a source media channel stream on the left pane and select execute
#                 to trigger multimodal checks.
#             </div>
#         </div>""", unsafe_allow_html=True)

#     if analyze_btn and uploaded_file is not None:

#         with tempfile.TemporaryDirectory() as tmpdir:
#             temp_video_path = os.path.join(tmpdir, "input_media.mp4")
#             temp_audio_path = os.path.join(tmpdir, "extracted_audio.wav")

#             with open(temp_video_path, "wb") as fh:
#                 fh.write(uploaded_file.read())

#             with st.spinner("Extracting audio channels and allocating thread vectors..."):
#                 audio_extracted = extract_audio(temp_video_path, temp_audio_path)

#             with st.spinner("Processing simultaneous model layer diagnostics..."):
#                 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
#                     vf = pool.submit(v_eng.analyze, temp_video_path)
#                     af = pool.submit(a_eng.analyze, temp_audio_path if audio_extracted else "")
#                     video_res = vf.result()
#                     audio_res = af.result()

#                 extracted_text = audio_res.get('transcript', '')
#                 text_res       = t_eng.analyze(extracted_text)

#             # ── Unpack VIDEO ──────────────────────────────────────────────
#             v_label     = video_res.get('label',           'Real')
#             v_score     = float(video_res.get('score',     0.0))
#             v_conf      = float(video_res.get('confidence',0.0))
#             v_faces     = video_res.get('faces_detected',  0)
#             v_reason    = video_res.get('verdict_reason',  '—')
#             v_peak_t    = float(video_res.get('peak_frame_time', 0.0))
#             v_temporal  = video_res.get('temporal',        {})
#             v_spikes    = v_temporal.get('spikes',         [])
#             v_variance  = float(v_temporal.get('variance', 0.0))
#             v_consist   = float(v_temporal.get('temporal_score', 1.0))
#             # Frame-level sigmoid array (all_probs preferred; fall back to frame_scores)
#             v_all_probs = video_res.get('all_probs', video_res.get('frame_scores', []))
#             # Optional performance charts returned by the engine
#             v_roc_fig   = video_res.get('roc_fig',  None)   # matplotlib Figure
#             v_cm_fig    = video_res.get('cm_fig',   None)   # matplotlib Figure
#             # Model metadata JSON
#             v_meta      = _video_meta("model_weights/deepfake_vision")
#             v_best_auc  = float(v_meta.get('best_auc',       0.9916))
#             v_best_thr  = float(v_meta.get('best_threshold', 0.8504))

#             # ── Unpack AUDIO ──────────────────────────────────────────────
#             a_label      = audio_res.get('label',        'Real')
#             a_score      = float(audio_res.get('score',  0.0))
#             a_conf       = float(audio_res.get('confidence', 0.0))
#             a_prob_real  = float(audio_res.get('prob_real',  1.0 - a_score))
#             a_prob_fake  = float(audio_res.get('prob_fake',  a_score))
#             a_manip_segs = audio_res.get('manipulated_segments',  'None detected')
#             a_peak_time  = audio_res.get('peak_suspicion_time',   None)
#             a_susp_bands = audio_res.get('suspicious_freq_bands', None)
#             a_result_plot= audio_res.get('result_plot',           None)  # matplotlib Figure

#             # ── Unpack TEXT ───────────────────────────────────────────────
#             t_label      = text_res.get('label',      'Real')
#             t_conf       = float(text_res.get('confidence', 0.5))
#             t_prob_real  = float(text_res.get('prob_real',
#                                               1.0 - t_conf if t_label == 'Fake' else t_conf))
#             t_prob_fake  = float(text_res.get('prob_fake',
#                                               t_conf if t_label == 'Fake' else 1.0 - t_conf))
#             t_tokens     = text_res.get('tokens',           [])
#             t_importance = text_res.get('token_importance', [])

#             # ── Derived flags & CSS classes ───────────────────────────────
#             v_is_fake = v_label == 'Fake'
#             a_is_fake = a_label == 'Fake'
#             t_is_fake = t_label == 'Fake'

#             v_badge_cls = "sv-badge-fake" if v_is_fake else "sv-badge-real"
#             a_badge_cls = "sv-badge-fake" if a_is_fake else "sv-badge-real"
#             t_badge_cls = "sv-badge-fake" if t_is_fake else "sv-badge-real"
#             v_fill = "sv-fill-fake" if v_is_fake else "sv-fill-real"
#             a_fill = "sv-fill-fake" if a_is_fake else "sv-fill-real"
#             t_fill = "sv-fill-fake" if t_is_fake else "sv-fill-real"

#             v_spike_cls  = "sv-diag-val-warn" if v_spikes    else "sv-diag-val"
#             v_var_cls    = "sv-diag-val-warn" if v_variance > 0.05 else "sv-diag-val"
#             a_peak_str   = f"{a_peak_time:.2f}s" if isinstance(a_peak_time, (int, float)) \
#                            else (str(a_peak_time) if a_peak_time else "—")
#             a_bands_str  = str(a_susp_bands) if a_susp_bands else "—"
#             a_seg_cls    = "sv-diag-val-warn" if str(a_manip_segs) not in ("None detected","") \
#                            else "sv-diag-val"

#             # =============================================================
#             # PANEL 1 — VIDEO FORENSIC ANALYSIS
#             # =============================================================
#             v_bar = _bar("Classification confidence", v_conf * 100, v_fill,
#                          "Sigmoid output probability that this video contains deepfake manipulation.")

#             st.markdown(f"""
#             <div class="sv-panel">
#                 <div class="sv-panel-header">
#                     <div>
#                         <div class="sv-panel-name">Video Core Visual Analysis</div>
#                         <div class="sv-panel-model">EfficientNet-B0 · Spatial Frame Ensemble · GeM Pooling · AUC {v_best_auc:.4f}</div>
#                     </div>
#                     <span class="sv-badge {v_badge_cls}">{v_label.upper()}</span>
#                 </div>

#                 <div class="sv-metric-row">
#                     <div class="sv-metric-card">
#                         <div class="sv-metric-label">Sigmoid Score</div>
#                         <div class="sv-metric-value">{v_score:.4f}</div>
#                         <div class="sv-metric-sub">all_probs mean</div>
#                     </div>
#                     <div class="sv-metric-card">
#                         <div class="sv-metric-label">Face Frames</div>
#                         <div class="sv-metric-value">{v_faces} / 16</div>
#                         <div class="sv-metric-sub">MTCNN detected</div>
#                     </div>
#                     <div class="sv-metric-card">
#                         <div class="sv-metric-label">Peak Anomaly</div>
#                         <div class="sv-metric-value">{v_peak_t:.1f}s</div>
#                         <div class="sv-metric-sub">highest-score frame</div>
#                     </div>
#                     <div class="sv-metric-card">
#                         <div class="sv-metric-label">Temporal Consistency</div>
#                         <div class="sv-metric-value">{v_consist:.2f}</div>
#                         <div class="sv-metric-sub">1.0 = stable</div>
#                     </div>
#                 </div>

#                 {v_bar}

#                 <div class="sv-section-label">Engine Diagnostics  (model_metadata.json)</div>
#                 <div style="background:#091525;border:1px solid #162a45;border-radius:6px;padding:10px 14px;">
#                     <div class="sv-diag-row">
#                         <span class="sv-diag-key">Peak Validation AUC</span>
#                         <span class="sv-diag-val">{v_best_auc:.4f}</span>
#                     </div>
#                     <div class="sv-diag-row">
#                         <span class="sv-diag-key">Optimal PR-F1 Threshold</span>
#                         <span class="sv-diag-val">{v_best_thr:.4f}</span>
#                     </div>
#                     <div class="sv-diag-row">
#                         <span class="sv-diag-key">Architecture</span>
#                         <span class="sv-diag-val">{v_meta.get('architecture','EfficientNet-B0 + GeM')}</span>
#                     </div>
#                     <div class="sv-diag-row">
#                         <span class="sv-diag-key">Training frames</span>
#                         <span class="sv-diag-val">{v_meta.get('training_frames',137990):,}</span>
#                     </div>
#                     <div class="sv-diag-row">
#                         <span class="sv-diag-key">Frame-score variance</span>
#                         <span class="{v_var_cls}">{v_variance:.6f}</span>
#                     </div>
#                     <div class="sv-diag-row">
#                         <span class="sv-diag-key">Spike frames</span>
#                         <span class="{v_spike_cls}">{str(v_spikes) if v_spikes else "none"}</span>
#                     </div>
#                     <div class="sv-diag-row">
#                         <span class="sv-diag-key">Verdict trace</span>
#                         <span class="sv-diag-val" style="font-size:10px;max-width:58%;text-align:right;
#                               white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{v_reason}</span>
#                     </div>
#                 </div>
#             </div>
#             """, unsafe_allow_html=True)

#             # Frame-level all_probs bar chart
#             if v_all_probs:
#                 st.markdown(
#                     "<div class='sv-section-label' style='margin-top:0;'>"
#                     "Frame-level sigmoid scores  (all_probs array)</div>",
#                     unsafe_allow_html=True
#                 )
#                 fig_fp, ax = plt.subplots(figsize=(7, 1.7))
#                 fig_fp.patch.set_facecolor('#091525')
#                 ax.set_facecolor('#091525')
#                 xs     = list(range(len(v_all_probs)))
#                 cols   = ['#ef4444' if s > v_eng.threshold else '#10b981' for s in v_all_probs]
#                 ax.bar(xs, v_all_probs, color=cols, width=0.7, zorder=3)
#                 ax.axhline(v_eng.threshold, color='#7ba4d0', linestyle='--', lw=1,
#                            label=f'threshold {v_eng.threshold:.2f}')
#                 ax.set_ylim(0, 1); ax.set_xlim(-0.5, len(v_all_probs) - 0.5)
#                 ax.tick_params(colors='#5a8ab8', labelsize=8)
#                 ax.grid(axis='y', color='#162a45', linewidth=0.5, zorder=0)
#                 for sp in ax.spines.values(): sp.set_edgecolor('#1a3050')
#                 ax.legend(fontsize=7, facecolor='#0d1e35', edgecolor='#1a3050', labelcolor='#7ba4d0')
#                 plt.tight_layout()
#                 st.pyplot(fig_fp, use_container_width=True)
#                 plt.close(fig_fp)

#             # ROC + Confusion Matrix expander
#             with st.expander("📊  Analytics & Performance Report  (ROC · Confusion Matrix)", expanded=False):
#                 c1, c2 = st.columns(2)
#                 with c1:
#                     if v_roc_fig is not None:
#                         st.pyplot(v_roc_fig, use_container_width=True)
#                     else:
#                         st.markdown("""
#                         <div class="sv-placeholder">
#                             ROC curve not returned by engine.<br>
#                             Set <code style='color:#5a8ab8'>roc_fig</code>
#                             in <code style='color:#5a8ab8'>VideoAnalyzer.analyze()</code>.
#                         </div>""", unsafe_allow_html=True)
#                 with c2:
#                     if v_cm_fig is not None:
#                         st.pyplot(v_cm_fig, use_container_width=True)
#                     else:
#                         st.markdown("""
#                         <div class="sv-placeholder">
#                             Confusion matrix not returned by engine.<br>
#                             Set <code style='color:#5a8ab8'>cm_fig</code>
#                             in <code style='color:#5a8ab8'>VideoAnalyzer.analyze()</code>.
#                         </div>""", unsafe_allow_html=True)

#             # =============================================================
#             # PANEL 2 — AUDIO FORENSIC ANALYSIS
#             # =============================================================
#             a_bar = _bar("Acoustic authenticity confidence", a_conf * 100, a_fill,
#                          "Whisper encoder classification certainty for this audio track.")

#             st.markdown(f"""
#             <div class="sv-panel">
#                 <div class="sv-panel-header">
#                     <div>
#                         <div class="sv-panel-name">Audio Frequency Track Analysis</div>
#                         <div class="sv-panel-model">Whisper-small Encoder · 8-Dialect ArFake · Gradient Saliency</div>
#                     </div>
#                     <span class="sv-badge {a_badge_cls}">{a_label.upper()}</span>
#                 </div>

#                 <div class="sv-metric-row">
#                     <div class="sv-metric-card">
#                         <div class="sv-metric-label">Raw Score</div>
#                         <div class="sv-metric-value">{a_score:.4f}</div>
#                         <div class="sv-metric-sub">softmax fake logit</div>
#                     </div>
#                     <div class="sv-metric-card">
#                         <div class="sv-metric-label">Confidence</div>
#                         <div class="sv-metric-value">{a_conf*100:.1f}%</div>
#                         <div class="sv-metric-sub">verdict certainty</div>
#                     </div>
#                     <div class="sv-metric-card">
#                         <div class="sv-metric-label">Anomaly Segments</div>
#                         <div class="sv-metric-value" style="font-size:14px;">{a_manip_segs}</div>
#                         <div class="sv-metric-sub">suspicious windows</div>
#                     </div>
#                 </div>

#                 <div class="sv-prob-row">
#                     <div class="sv-prob-chip sv-prob-chip-real">
#                         <div class="sv-prob-chip-label">P(Real)</div>
#                         <div class="sv-prob-chip-val">{a_prob_real:.4f}</div>
#                     </div>
#                     <div class="sv-prob-chip sv-prob-chip-fake">
#                         <div class="sv-prob-chip-label">P(Fake)</div>
#                         <div class="sv-prob-chip-val">{a_prob_fake:.4f}</div>
#                     </div>
#                 </div>

#                 {a_bar}

#                 <div class="sv-section-label">Computed Diagnostic Triggers</div>
#                 <div style="background:#091525;border:1px solid #162a45;border-radius:6px;padding:10px 14px;">
#                     <div class="sv-diag-row">
#                         <span class="sv-diag-key">Peak suspicion time</span>
#                         <span class="{"sv-diag-val-warn" if a_is_fake else "sv-diag-val"}">{a_peak_str}</span>
#                     </div>
#                     <div class="sv-diag-row">
#                         <span class="sv-diag-key">Suspicious frequency bands</span>
#                         <span class="{"sv-diag-val-warn" if a_susp_bands else "sv-diag-val"}">{a_bands_str}</span>
#                     </div>
#                     <div class="sv-diag-row">
#                         <span class="sv-diag-key">Manipulated segments</span>
#                         <span class="{a_seg_cls}">{a_manip_segs}</span>
#                     </div>
#                 </div>
#             </div>
#             """, unsafe_allow_html=True)

#             # Matplotlib dashboard (waveform + saliency + spectrogram)
#             if a_result_plot is not None:
#                 st.markdown(
#                     "<div class='sv-section-label'>Waveform · Saliency Curve · Spectrogram</div>",
#                     unsafe_allow_html=True
#                 )
#                 st.pyplot(a_result_plot, use_container_width=True)
#             else:
#                 st.markdown("""
#                 <div class="sv-placeholder">
#                     Waveform / saliency dashboard not returned by audio engine.<br>
#                     Set <code style='color:#5a8ab8'>result_plot</code>
#                     (matplotlib Figure) in <code style='color:#5a8ab8'>AudioAnalyzer.analyze()</code>.
#                 </div>""", unsafe_allow_html=True)

#             # =============================================================
#             # PANEL 3 — TEXT SEMANTIC ANALYSIS
#             # =============================================================
#             t_bar = _bar("Information authenticity prediction", t_conf * 100, t_fill,
#                          "AraBERT classification confidence across 606,912 AFND article embeddings.")

#             token_html = _token_highlights(t_tokens, t_importance)

#             st.markdown(f"""
#             <div class="sv-panel">
#                 <div class="sv-panel-header">
#                     <div>
#                         <div class="sv-panel-name">Arabic Text Semantic Analysis</div>
#                         <div class="sv-panel-model">AraBERT v02 · AFND 606k Articles · Gradient × Embedding Norm</div>
#                     </div>
#                     <span class="sv-badge {t_badge_cls}">{t_label.upper()}</span>
#                 </div>

#                 <div class="sv-prob-row">
#                     <div class="sv-prob-chip sv-prob-chip-real">
#                         <div class="sv-prob-chip-label">P(Real)</div>
#                         <div class="sv-prob-chip-val">{t_prob_real:.4f}</div>
#                     </div>
#                     <div class="sv-prob-chip sv-prob-chip-fake">
#                         <div class="sv-prob-chip-label">P(Fake)</div>
#                         <div class="sv-prob-chip-val">{t_prob_fake:.4f}</div>
#                     </div>
#                 </div>

#                 {t_bar}

#                 <div class="sv-section-label">Token Importance Map  (gradient × embedding norm)</div>
#                 <div style="background:#091525;border:1px solid #1a3050;border-radius:6px;
#                             padding:12px 14px;max-height:200px;overflow-y:auto;">
#                     {token_html}
#                 </div>
#                 <div style="margin-top:6px;font-size:10px;color:#3d6a99;line-height:1.5;">
#                     Colour intensity = influence on classification.
#                     <span style="color:#ef4444;">■</span> High &nbsp;·&nbsp;
#                     <span style="color:#1a3050;border:1px solid #2a4a6a;padding:0 3px;">■</span> Low
#                 </div>

#                 <div class="sv-section-label">Extracted Speech Transcript (Arabic Language Context)</div>
#             </div>
#             """, unsafe_allow_html=True)

#             # Transcript rendered in its own container for scroll isolation
#             st.markdown(
#                 f'<div class="sv-transcript-box">{extracted_text or "No transcript extracted."}</div>',
#                 unsafe_allow_html=True
#             )

"""
inference_engines/video_engine.py
Video deepfake detection engine — EfficientNet-B0 + GeM head.

Public interface expected by app.py:
    VideoAnalyzer(model_path: str, threshold: float = 0.72)
        .threshold   -- settable, used at inference time
        .news_mode   -- settable (informational flag)
        .process_video(video_path: str) -> dict with keys:
            score, label, confidence, faces_detected, verdict_reason,
            peak_frame_time, all_probs, temporal {variance, spikes,
            temporal_score}, roc_fig, cm_fig, video_path
"""

import os
import json
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from facenet_pytorch import MTCNN

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GeM(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return F.adaptive_avg_pool2d(x.clamp(min=self.eps).pow(self.p), 1).pow(1.0 / self.p)


class VideoAnalyzer:
    def __init__(self, model_path, threshold=0.72, news_mode=False, n_frames=16):
        self.device    = DEVICE
        self.threshold = threshold
        self.news_mode = news_mode
        self.n_frames  = n_frames
        self.model_path = model_path

        self.model = self._build_model()

        if os.path.exists(model_path):
            state = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state['model_state'] if 'model_state' in state else state)
        else:
            # Allows the Streamlit app to boot (e.g. for UI work) before the
            # trained checkpoint has been placed in model_weights/deepfake_vision/.
            print(f"[VideoAnalyzer] WARNING: checkpoint not found at '{model_path}'. "
                  f"Running with randomly-initialized weights.")

        self.model.eval().to(self.device)

        self.mtcnn = MTCNN(keep_all=False, device=self.device)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self._metadata = self._load_metadata()

    # ── Model definition ──────────────────────────────────────────────────
    def _build_model(self):
        model = models.efficientnet_b0(weights=None)
        model.avgpool = GeM()
        model.classifier = nn.Sequential(
            nn.Flatten(), nn.BatchNorm1d(1280), nn.Dropout(0.4),
            nn.Linear(1280, 256), nn.SiLU(), nn.BatchNorm1d(256),
            nn.Dropout(0.3), nn.Linear(256, 1)
        )
        return model

    # ── Metadata (for ROC / confusion matrix figures) ───────────────────────
    def _load_metadata(self):
        meta_path = os.path.join(os.path.dirname(self.model_path), "model_metadata.json")
        if os.path.exists(meta_path):
            try:
                with open(meta_path) as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                return {}
        return {}

    # ── Main entry point ────────────────────────────────────────────────────
    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total <= 0:
            cap.release()
            return self._empty_result(video_path, "Unreadable video / zero frames")

        indices = np.linspace(0, max(total - 1, 0), self.n_frames, dtype=int)

        faces, face_frame_nums = [], []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret:
                continue
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face = self.mtcnn(frame_rgb)
            if face is not None:
                faces.append(face)
                face_frame_nums.append(int(idx))
        cap.release()

        if not faces:
            return self._empty_result(video_path, "No face detected in sampled frames")

        # Per-frame sigmoid scores (not just the mean) so we can derive
        # temporal consistency, spikes, and the peak-anomaly timestamp.
        batch = torch.stack(faces).to(self.device)
        with torch.no_grad():
            all_probs = torch.sigmoid(self.model(batch)).squeeze(-1).cpu().numpy()
        all_probs = np.atleast_1d(all_probs).astype(float)

        score = float(all_probs.mean())
        label = 'Real' if score >= self.threshold else 'Fake'
        confidence = float(min(abs(score - self.threshold) * 2.0, 1.0))

        peak_idx          = int(np.argmax(all_probs))
        peak_frame_number = face_frame_nums[peak_idx]
        peak_frame_time   = peak_frame_number / fps if fps > 0 else 0.0

        variance = float(np.var(all_probs))
        std      = float(all_probs.std())
        spike_thresh = all_probs.mean() + (2 * std if std > 0 else 0.15)
        spikes = [
            round(face_frame_nums[i] / fps, 2)
            for i, p in enumerate(all_probs)
            if p > spike_thresh
        ]
        # Heuristic: low frame-to-frame variance -> high temporal consistency.
        temporal_score = float(max(0.0, 1.0 - variance * 4))

        roc_fig, cm_fig = self._build_analytics_figures()

        return {
            'score':           score,
            'label':           label,
            'confidence':      confidence,
            'faces_detected':  len(faces),
            'verdict_reason':  f"mean_score={score:.3f} vs threshold={self.threshold:.2f} "
                                f"({len(faces)}/{self.n_frames} frames with faces)",
            'peak_frame_time': peak_frame_time,
            'all_probs':       all_probs.tolist(),
            'temporal': {
                'variance':       variance,
                'spikes':         spikes,
                'temporal_score': temporal_score,
            },
            'roc_fig':  roc_fig,
            'cm_fig':   cm_fig,
            'video_path': video_path,
        }

    # ── Fallback result (no readable frames / no faces) ─────────────────────
    def _empty_result(self, video_path, reason):
        roc_fig, cm_fig = self._build_analytics_figures()
        return {
            'score':           0.5,
            'label':           'Fake',
            'confidence':      0.0,
            'faces_detected':  0,
            'verdict_reason':  reason,
            'peak_frame_time': 0.0,
            'all_probs':       [],
            'temporal': {'variance': 0.0, 'spikes': [], 'temporal_score': 1.0},
            'roc_fig':  roc_fig,
            'cm_fig':   cm_fig,
            'video_path': video_path,
        }

    # ── Optional analytics figures, themed to match the UI ──────────────────
    def _build_analytics_figures(self):
        roc_data = self._metadata.get('roc_curve')
        cm_data  = self._metadata.get('confusion_matrix')

        roc_fig = None
        if roc_data and 'fpr' in roc_data and 'tpr' in roc_data:
            fpr, tpr = roc_data['fpr'], roc_data['tpr']
            auc = self._metadata.get('best_auc')
            roc_fig, ax = plt.subplots(figsize=(4, 3.2))
            roc_fig.patch.set_facecolor('#0d1e35')
            ax.set_facecolor('#0d1e35')
            label = f"AUC = {auc:.4f}" if auc is not None else "Model ROC"
            ax.plot(fpr, tpr, color='#10b981', lw=2, label=label)
            ax.plot([0, 1], [0, 1], color='#3d6a99', lw=1, linestyle='--')
            ax.set_xlabel("False Positive Rate", color='#5a8ab8', fontsize=9)
            ax.set_ylabel("True Positive Rate", color='#5a8ab8', fontsize=9)
            ax.set_title("ROC Curve", color='#e8f0fa', fontsize=10)
            ax.tick_params(colors='#5a8ab8', labelsize=8)
            for sp in ax.spines.values():
                sp.set_edgecolor('#1a3050')
            ax.legend(fontsize=8, facecolor='#0d1e35', edgecolor='#1a3050', labelcolor='#7ba4d0')
            plt.tight_layout()

        cm_fig = None
        if cm_data:
            cm = np.array(cm_data)
            cm_fig, ax = plt.subplots(figsize=(4, 3.2))
            cm_fig.patch.set_facecolor('#0d1e35')
            ax.set_facecolor('#0d1e35')
            ax.imshow(cm, cmap='Blues')
            ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
            ax.set_xticklabels(['Real', 'Fake'], color='#5a8ab8')
            ax.set_yticklabels(['Real', 'Fake'], color='#5a8ab8')
            ax.set_xlabel("Predicted", color='#5a8ab8', fontsize=9)
            ax.set_ylabel("Actual", color='#5a8ab8', fontsize=9)
            ax.set_title("Confusion Matrix", color='#e8f0fa', fontsize=10)
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                            color='#0a1628', fontweight='bold')
            for sp in ax.spines.values():
                sp.set_edgecolor('#1a3050')
            plt.tight_layout()

        return roc_fig, cm_fig
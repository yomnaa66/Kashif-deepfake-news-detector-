
"""
inference_engines/text_engine.py
AraBERT v02 Text Sequence Normalization & Fake News Content Classifier

Compatible with app.py:
  - TextAnalyzer(model_dir)
  - .analyze(text_content) -> dict with keys:
        label, confidence, score, prob_real, prob_fake,
        tokens, token_importance, cleaned_text
"""

import os
import re
import torch
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Dict, Any, List, Tuple

DEVICE  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAX_LEN = 512  # Match training MAX_LEN for proper tokenization


def _remove_urls(text):
    return re.sub(r'https?://\S+|www\.\S+', ' ', text)


def _remove_html(text):
    return re.sub(r'<[^>]+>', ' ', text)


def _remove_emails(text):
    return re.sub(r'\S+@\S+\.\S+', ' ', text)


def _remove_phones(text):
    return re.sub(r'(\+?\d[\d\s\-]{7,}\d)', ' ', text)


def _remove_emojis(text):
    return re.sub(
        r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF'
        r'\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF'
        r'\U00002702-\U000027B0\U000024C2-\U0001F251]+',
        ' ', text, flags=re.UNICODE
    )


def _remove_non_arabic(text):
    return re.sub(
        r'[^\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF\s\d]',
        ' ', text
    )


def _normalize_arabic(text):
    text = re.sub(r'[\u0622\u0623\u0625\u0671]', '\u0627', text)
    text = re.sub(r'\u0629', '\u0647', text)
    text = re.sub(r'\u0649', '\u064a', text)
    text = re.sub(r'[\u064B-\u065F]', '', text)
    text = re.sub(r'\u0640', '', text)
    text = re.sub(r'(.)\1{3,}', r'\1\1', text)
    return text


def _clean_spaces(text):
    return re.sub(r'\s+', ' ', text).strip()


def preprocess_arabic(text):
    if not isinstance(text, str):
        return ''
    text = _remove_urls(text)
    text = _remove_html(text)
    text = _remove_emails(text)
    text = _remove_phones(text)
    text = _remove_emojis(text)
    # Don't remove non-Arabic - keep English/mixed languages for model
    # text = _remove_non_arabic(text)  
    text = _normalize_arabic(text)
    text = _clean_spaces(text)
    return text


def _get_arabert_preprocessor():
    try:
        from arabert.preprocess import ArabertPreprocessor
        return ArabertPreprocessor(model_name="bert-base-arabertv02")
    except ImportError:
        return None


def _compute_token_importance(model, tokenizer, cleaned_text, device):
    if not cleaned_text.strip():
        return [], []

    try:
        inputs = tokenizer(
            cleaned_text,
            max_length=MAX_LEN,
            padding=True,
            truncation=True,
            return_tensors='pt',
        ).to(device)

        embedding_layer = model.get_input_embeddings()
        input_ids       = inputs['input_ids']

        embeddings = embedding_layer(input_ids)
        embeddings.retain_grad()

        model.zero_grad()

        outputs = model(
            inputs_embeds=embeddings,
            attention_mask=inputs.get('attention_mask'),
            token_type_ids=inputs.get('token_type_ids'),
        )

        predicted_class = int(outputs.logits.argmax(dim=-1).item())
        outputs.logits[0, predicted_class].backward()

        if embeddings.grad is None:
            raise RuntimeError("No gradient on embeddings")

        grad_emb       = (embeddings.grad * embeddings).detach().cpu().numpy()
        importance_raw = np.linalg.norm(grad_emb[0], axis=-1)

        token_ids = input_ids[0].cpu().tolist()
        attention  = inputs['attention_mask'][0].cpu().tolist()

        tokens = []
        scores = []
        for tok_id, attn, imp in zip(token_ids, attention, importance_raw):
            if attn == 0:
                break
            tok_str = tokenizer.convert_ids_to_tokens(tok_id)
            if tok_str in (tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token):
                continue
            tokens.append(tok_str)
            scores.append(float(imp))

        if scores:
            mn, mx = min(scores), max(scores)
            importance = [(s - mn) / (mx - mn + 1e-8) for s in scores]
        else:
            importance = []

        return tokens, importance

    except Exception:
        words = cleaned_text.split()[:64]
        return words, [0.5] * len(words)


class TextAnalyzer:

    # [Enhancement] FIX: 2 words is not enough signal for a transformer
    # classifier — short fragments (common from noisy transcripts) were
    # producing confident-looking but meaningless scores. Raised to 5,
    # matching the threshold used by the model during training/eval.
    SHORT_THRESHOLD = 5

    def __init__(self, model_dir):
        self.model_base_name = 'aubmindlab/bert-base-arabertv02'
        self.tokenizer       = None
        self.model           = None
        self._model_dir      = model_dir
        self._arabert_prep   = _get_arabert_preprocessor()
        # [Enhancement] FIX: track whether we actually loaded a fine-tuned
        # fake-news checkpoint, vs falling back to the un-finetuned base
        # model (which has a randomly-initialized classification head and
        # produces near-coin-flip, meaningless scores while *looking* like
        # a real verdict). analyze() uses this to be honest about it.
        self._using_finetuned = False

        try:
            has_finetuned_ckpt = bool(
                model_dir and os.path.exists(os.path.join(model_dir, "config.json"))
            )
            load_target = model_dir if has_finetuned_ckpt else self.model_base_name
            self.tokenizer = AutoTokenizer.from_pretrained(load_target)
            self.model     = AutoModelForSequenceClassification.from_pretrained(load_target)
            self.model.eval().to(DEVICE)
            self._using_finetuned = has_finetuned_ckpt
        except Exception as e:
            print("Text init error:", e)
            self.tokenizer = None
            self.model     = None

    def _full_preprocess(self, text_content):
        raw = text_content
        if self._arabert_prep is not None:
            try:
                raw = self._arabert_prep.preprocess(raw)
            except Exception:
                pass
        cleaned_text = preprocess_arabic(raw)
        return cleaned_text

    def _resolve_labels(self, probabilities):
        id2label = getattr(self.model.config, 'id2label', {0: 'Fake', 1: 'Real'})

        prob_map = {}
        for idx, lbl in id2label.items():
            idx = int(idx)
            if idx < len(probabilities):
                prob_map[lbl.capitalize()] = float(probabilities[idx])

        prob_real = prob_map.get('Real', 0.5)
        prob_fake = prob_map.get('Fake', 0.5)

        label      = 'Real' if prob_real >= 0.50 else 'Fake'
        confidence = prob_real if label == 'Real' else prob_fake
        return label, confidence, prob_real, prob_fake

    def analyze(self, text_content):
        cleaned_text = ''

        if not text_content or not isinstance(text_content, str) or not text_content.strip():
            return {
                'label':            'Real',
                'confidence':       0.50,
                'score':            0.50,
                'prob_real':        0.50,
                'prob_fake':        0.50,
                'tokens':           [],
                'token_importance': [],
                'cleaned_text':     cleaned_text,
            }

        cleaned_text = self._full_preprocess(text_content)

        if len(cleaned_text.split()) < self.SHORT_THRESHOLD:
            return {
                'label':            'Real',
                'confidence':       0.50,
                'score':            0.50,
                'prob_real':        0.50,
                'prob_fake':        0.50,
                'tokens':           [],
                'token_importance': [],
                'cleaned_text':     cleaned_text,
            }

        if self.model is None or self.tokenizer is None:
            words = cleaned_text.split()[:64]
            return {
                'label':            'Real',
                'confidence':       0.50,
                'score':            0.50,
                'prob_real':        0.50,
                'prob_fake':        0.50,
                'tokens':           words,
                'token_importance': [0.5] * len(words),
                'cleaned_text':     cleaned_text,
            }

        # [Enhancement] FIX: running on the un-finetuned base model produces
        # a confident-looking but meaningless score (random classification
        # head). Be honest about it instead of returning a fake verdict.
        if not self._using_finetuned:
            words = cleaned_text.split()[:64]
            return {
                'label':            'Real',
                'confidence':       0.50,
                'score':            0.50,
                'prob_real':        0.50,
                'prob_fake':        0.50,
                'tokens':           words,
                'token_importance': [0.5] * len(words),
                'cleaned_text':     cleaned_text,
            }

        try:
            inputs = self.tokenizer(
                cleaned_text,
                max_length=MAX_LEN,
                padding=True,
                truncation=True,
                return_tensors='pt',
            ).to(DEVICE)

            with torch.no_grad():
                outputs       = self.model(**inputs)
                probabilities = (
                    torch.softmax(outputs.logits, dim=-1)
                    .cpu().numpy().flatten()
                )

            label, confidence, prob_real, prob_fake = self._resolve_labels(probabilities)

            self.model.train()
            try:
                tokens, token_importance = _compute_token_importance(
                    self.model, self.tokenizer, cleaned_text, DEVICE
                )
            finally:
                self.model.eval()

            return {
                'label':            label,
                'confidence':       confidence,
                'score':            prob_real,
                'prob_real':        prob_real,
                'prob_fake':        prob_fake,
                'tokens':           tokens,
                'token_importance': token_importance,
                'cleaned_text':     cleaned_text,
            }

        except Exception as e:
            return {
                'label':            'Error',
                'confidence':       0.50,
                'score':            0.50,
                'prob_real':        0.50,
                'prob_fake':        0.50,
                'tokens':           [],
                'token_importance': [],
                'cleaned_text':     f"Text execution runtime failure: {str(e)}",
            }
# """
# inference_engines/text_engine.py
# AraBERT v02 Text Sequence Normalization & Fake News Content Classifier

# Returns (per app.py contract):
#     label            : str   – 'Real' | 'Fake'
#     confidence       : float – confidence of the winning class (0–1)
#     score            : float – prob_real (kept for legacy callers)
#     prob_real        : float – P(Real)
#     prob_fake        : float – P(Fake)
#     cleaned_text     : str
#     tokens           : list[str]   – sub-word tokens (no special tokens)
#     token_importance : list[float] – gradient × embedding-norm saliency
# """

# import os
# import re
# import torch
# import torch.nn as nn
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# from typing import Dict, Any, List

# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# MODEL_BASE_NAME = 'aubmindlab/bert-base-arabertv02'

# # ── Optional official AraBERT preprocessor (matches training pipeline) ───────
# ARABERT_PREPROCESSOR_AVAILABLE = False
# _arabert_prep = None
# try:
#     from arabert.preprocess import ArabertPreprocessor
#     _arabert_prep = ArabertPreprocessor(model_name=MODEL_BASE_NAME)
#     ARABERT_PREPROCESSOR_AVAILABLE = True
# except Exception:
#     _arabert_prep = None
#     ARABERT_PREPROCESSOR_AVAILABLE = False


# # ── Text normalisation (must EXACTLY match training-time Section 7) ─────────

# def _remove_urls(text: str) -> str:
#     return re.sub(r'https?://\S+|www\.\S+', ' ', text)

# def _remove_html(text: str) -> str:
#     return re.sub(r'<[^>]+>', ' ', text)

# def _remove_emails(text: str) -> str:
#     return re.sub(r'\S+@\S+\.\S+', ' ', text)

# def _remove_phones(text: str) -> str:
#     return re.sub(r'(\+?\d[\d\s\-]{7,}\d)', ' ', text)

# def _remove_emojis(text: str) -> str:
#     return re.sub(
#         r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF'
#         r'\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF'
#         r'\U00002702-\U000027B0\U000024C2-\U0001F251]+',
#         ' ', text, flags=re.UNICODE
#     )

# def _remove_non_arabic(text: str) -> str:
#     # Keep Arabic unicode blocks, digits, whitespace
#     return re.sub(
#         r'[^\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF\s\d]',
#         ' ', text
#     )

# def _normalize_arabic(text: str) -> str:
#     text = re.sub(r'[\u0622\u0623\u0625\u0671]', '\u0627', text)  # Alef variants → ا
#     text = re.sub(r'\u0629', '\u0647', text)                       # ة → ه
#     text = re.sub(r'\u0649', '\u064a', text)                       # ى → ي
#     text = re.sub(r'[\u064B-\u065F]', '', text)                    # Remove tashkeel/diacritics
#     text = re.sub(r'\u0640', '', text)                             # Remove tatweel
#     text = re.sub(r'(.)\1{3,}', r'\1\1', text)                     # Collapse repeated chars (≥4 → 2)
#     return text

# def _clean_spaces(text: str) -> str:
#     return re.sub(r'\s+', ' ', text).strip()


# def preprocess_arabic(text: str) -> str:
#     """
#     Reproduces the exact cleaning pipeline applied to training data
#     (notebook Section 7) so that inference-time inputs match the
#     distribution the model was trained on.
#     """
#     if not isinstance(text, str):
#         return ''

#     text = _remove_urls(text)
#     text = _remove_html(text)
#     text = _remove_emails(text)
#     text = _remove_phones(text)
#     text = _remove_emojis(text)
#     text = _remove_non_arabic(text)
#     text = _normalize_arabic(text)
#     text = _clean_spaces(text)

#     if ARABERT_PREPROCESSOR_AVAILABLE and _arabert_prep is not None:
#         try:
#             text = _arabert_prep.preprocess(text)
#         except Exception:
#             pass

#     return text


# # ── Saliency helper ───────────────────────────────────────────────────────────

# def _gradient_embedding_saliency(
#     model:     AutoModelForSequenceClassification,
#     tokenizer: AutoTokenizer,
#     inputs:    dict,
#     target_idx: int,
# ) -> List[float]:
#     """
#     Computes gradient × embedding-norm saliency for each input token.
#     Falls back to a uniform list on any error.
#     """
#     n_tokens = inputs['input_ids'].shape[1]

#     try:
#         # Obtain the embedding layer; works for BERT-family architectures
#         embed_layer = model.bert.embeddings.word_embeddings

#         input_ids = inputs['input_ids']
#         emb = embed_layer(input_ids)          # (1, seq, hidden)
#         emb.retain_grad()

#         # Forward pass through the rest of the model using embeddings directly
#         attention_mask  = inputs.get('attention_mask')
#         token_type_ids  = inputs.get('token_type_ids')

#         outputs = model(
#             inputs_embeds   = emb,
#             attention_mask  = attention_mask,
#             token_type_ids  = token_type_ids,
#         )

#         logit = outputs.logits[0, target_idx]
#         model.zero_grad()
#         logit.backward()

#         # gradient × L2-norm of embeddings → per-token scalar
#         grad  = emb.grad[0]                   # (seq, hidden)
#         norms = (grad * emb[0]).norm(dim=-1)  # (seq,)
#         saliency = norms.detach().cpu().tolist()
#         return saliency

#     except Exception:
#         return [0.0] * n_tokens


# # ── Analyser class ────────────────────────────────────────────────────────────

# class TextAnalyzer:
#     def __init__(self, model_dir: str):
#         self.model_base_name = MODEL_BASE_NAME
#         self.tokenizer = None
#         self.model     = None

#         try:
#             if model_dir and os.path.exists(os.path.join(model_dir, "config.json")):
#                 load_target = model_dir
#             else:
#                 load_target = self.model_base_name

#             self.tokenizer = AutoTokenizer.from_pretrained(load_target)
#             self.model     = AutoModelForSequenceClassification.from_pretrained(load_target)
#             self.model.to(DEVICE)
#             # Keep in train mode so gradients flow during saliency pass
#         except Exception:
#             self.tokenizer = None
#             self.model     = None

#     # ── Public interface ──────────────────────────────────────────────────────

#     def analyze(self, text_content: str) -> Dict[str, Any]:
#         """
#         Classify text as Real/Fake and return the full dict consumed by app.py.

#         Keys returned
#         -------------
#         label            : 'Real' | 'Fake'
#         confidence       : float 0-1
#         score            : float (== prob_real, kept for legacy callers)
#         prob_real        : float
#         prob_fake        : float
#         cleaned_text     : str
#         tokens           : list[str]
#         token_importance : list[float]
#         """
#         _fallback_real = {
#             'label': 'Real', 'score': 0.50, 'confidence': 0.50,
#             'prob_real': 0.50, 'prob_fake': 0.50,
#             'cleaned_text': '', 'tokens': [], 'token_importance': [],
#         }
#         _fallback_fake = {**_fallback_real, 'label': 'Fake'}

#         if not text_content or len(text_content.strip()) < 5:
#             return _fallback_fake

#         cleaned_text = preprocess_arabic(text_content)

#         if self.model is None or self.tokenizer is None:
#             return {**_fallback_real, 'cleaned_text': cleaned_text}

#         try:
#             inputs = self.tokenizer(
#                 cleaned_text,
#                 max_length=128,
#                 padding='max_length',
#                 truncation=True,
#                 return_tensors='pt',
#             )
#             inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

#             # ── Classification (no-grad) ──────────────────────────────────
#             self.model.eval()
#             with torch.no_grad():
#                 outputs = self.model(**inputs)
#                 probs   = torch.softmax(outputs.logits, dim=-1).cpu().numpy().flatten()

#             # Index mapping: 0 = Fake, 1 = Real  (per training convention)
#             prob_fake = float(probs[0])
#             prob_real = float(probs[1])

#             label      = 'Real' if prob_real >= 0.50 else 'Fake'
#             confidence = prob_real if label == 'Real' else prob_fake
#             target_idx = 1 if label == 'Real' else 0  # class to back-prop

#             # ── Token list (strip [CLS], [SEP], padding) ──────────────────
#             all_ids    = inputs['input_ids'][0].tolist()
#             all_tokens = self.tokenizer.convert_ids_to_tokens(all_ids)
#             pad_id     = self.tokenizer.pad_token_id
#             sep_id     = self.tokenizer.sep_token_id
#             cls_id     = self.tokenizer.cls_token_id
#             skip_ids   = {pid for pid in [pad_id, sep_id, cls_id] if pid is not None}

#             clean_tokens   = []
#             content_indices = []
#             for i, (tid, tok) in enumerate(zip(all_ids, all_tokens)):
#                 if tid not in skip_ids:
#                     clean_tokens.append(tok)
#                     content_indices.append(i)

#             # ── Saliency (gradient × embedding norm) ─────────────────────
#             self.model.train()   # enable grad tracking
#             raw_saliency = _gradient_embedding_saliency(
#                 self.model, self.tokenizer, inputs, target_idx
#             )
#             self.model.eval()

#             token_importance = [raw_saliency[i] for i in content_indices] \
#                                if raw_saliency else [0.0] * len(clean_tokens)

#             return {
#                 'label':            label,
#                 'score':            prob_real,
#                 'confidence':       confidence,
#                 'prob_real':        prob_real,
#                 'prob_fake':        prob_fake,
#                 'cleaned_text':     cleaned_text,
#                 'tokens':           clean_tokens,
#                 'token_importance': token_importance,
#             }

#         except Exception:
#             cleaned = preprocess_arabic(text_content) if text_content else ''
#             return {**_fallback_fake, 'cleaned_text': cleaned}

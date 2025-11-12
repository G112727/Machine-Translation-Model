# app.py
import os
import re
import logging
import torch

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ES_ID = r"J:\FINAL PROJECT\marian_bilingual_en_es_pt\en-es_best"
PT_ID = r"J:\FINAL PROJECT\marian_bilingual_en_es_pt\en-pt_best"

# built-in scientific glossary (you can extend this or load from CSV)
FALLBACK_GLOSSARY = [
    {
        "en": "polymerase chain reaction",
        "es": "reacción en cadena de la polimerasa",
        "pt": "reação em cadeia da polimerase",
    },
    {
        "en": "confidence interval",
        "es": "intervalo de confianza",
        "pt": "intervalo de confiança",
    },
]

# -------------------------------------------------
# LOGGING
# -------------------------------------------------
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("translator")

log.info(f"Device: {DEVICE}")

# -------------------------------------------------
# LOAD MODELS (once)
# -------------------------------------------------
log.info("Loading models…")
tok_es = AutoTokenizer.from_pretrained(ES_ID, use_fast=False)
mod_es = AutoModelForSeq2SeqLM.from_pretrained(ES_ID).to(DEVICE)
tok_pt = AutoTokenizer.from_pretrained(PT_ID, use_fast=False)
mod_pt = AutoModelForSeq2SeqLM.from_pretrained(PT_ID).to(DEVICE)
log.info("Models loaded.")

# -------------------------------------------------
# FASTAPI APP
# -------------------------------------------------
app = FastAPI(title="EN→ES/PT Translator (Marian)")
app.mount("/static", StaticFiles(directory="static"), name="static")


# -------------------------------------------------
# SCHEMAS
# -------------------------------------------------
class TranslateRequest(BaseModel):
    text: str
    target: str  # "es" or "pt"


class TranslateLongRequest(BaseModel):
    text: str
    target: str  # "es" or "pt"
    max_chars: int = 900
    max_new_tokens: int = 128
    num_beams: int = 5


# -------------------------------------------------
# HELPERS: text reading (for PDF/TXT) -- optional for API
# -------------------------------------------------
def read_text(path: str) -> str:
    if path.lower().endswith(".txt"):
        return open(path, "r", encoding="utf-8", errors="ignore").read()
    if path.lower().endswith(".pdf"):
        try:
            import fitz  # PyMuPDF
        except ImportError as e:
            raise RuntimeError("PDF reading failed. Install: pip install pymupdf") from e
        doc = fitz.open(path)
        texts = []
        for page in doc:
            texts.append(page.get_text("text"))
        return "\n".join(texts)
    raise ValueError("Unsupported file type. Provide .txt or .pdf")


# -------------------------------------------------
# HELPERS: protect / restore equations and citations
# -------------------------------------------------
EQ_RE = r'(\$[^$]+\$|\\\([^\)]+\\\)|\\\[[^\]]+\\\])'
CIT_RE = r'(\[[0-9,\-\s]+\])'

def protect_specials(text: str):
    # equations
    eqs = re.findall(EQ_RE, text)
    for i, m in enumerate(eqs, 1):
        text = text.replace(m, f"<EQ{i}>", 1)
    # citations
    cits = re.findall(CIT_RE, text)
    for i, m in enumerate(cits, 1):
        text = text.replace(m, f"<CIT{i}>", 1)
    rep = {"eqs": eqs, "cits": cits}
    return text, rep

def restore_specials(text: str, rep):
    for i, m in enumerate(rep.get("eqs", []), 1):
        text = text.replace(f"<EQ{i}>", m)
    for i, m in enumerate(rep.get("cits", []), 1):
        text = text.replace(f"<CIT{i}>", m)
    return text


# -------------------------------------------------
# HELPERS: glossary → placeholders → restore
# -------------------------------------------------
def build_placeholder_map_for_lang(text: str, glossary, lang: str):
    """
    Replace EN scientific terms in text with <G1>, <G2>, ... and remember
    what each placeholder should become in the target language.
    """
    # longest-first
    entries = sorted(glossary, key=lambda r: len(r["en"]), reverse=True)
    ph_map = {}
    gid = 1

    def make_pattern(term: str) -> re.Pattern:
        t = re.sub(r"\s*-\s*|\s+", r"[ -]+", re.escape(term.strip()))
        return re.compile(rf"(?i)\b{t}\b")

    for row in entries:
        en_term = row["en"]
        tgt_term = row.get(lang, "").strip()
        if not en_term or not tgt_term:
            continue
        pat = make_pattern(en_term)
        def _repl(m):
            nonlocal gid
            ph = f"<G{gid}>"
            gid += 1
            ph_map[ph] = tgt_term
            return ph
        text = pat.sub(_repl, text)
    return text, ph_map

def replace_placeholders(text: str, ph_map: dict) -> str:
    for ph in sorted(ph_map.keys(), key=len, reverse=True):
        text = text.replace(ph, ph_map[ph])
    return text


# -------------------------------------------------
# HELPERS: chunking
# -------------------------------------------------
def split_text(text: str, max_chars: int = 900):
    # paragraph blocks first
    paras = re.split(r'\n{2,}', text.strip())
    chunks = []
    for para in paras:
        para = para.strip()
        if not para:
            continue
        if len(para) <= max_chars:
            chunks.append(para)
        else:
            # split by sentence
            sentences = re.split(r'(?<=[.!?])\s+', para)
            buf = ""
            for s in sentences:
                if len(buf) + len(s) + 1 <= max_chars:
                    buf = (buf + " " + s).strip()
                else:
                    if buf:
                        chunks.append(buf)
                    buf = s
            if buf:
                chunks.append(buf)
    return chunks


# -------------------------------------------------
# CORE: single-shot generation (short text)
# -------------------------------------------------
@torch.inference_mode()
def generate(text: str, tok, model, max_new_tokens=128, num_beams=4):
    if not text.strip():
        return ""
    enc = tok(
        [text],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,   # a bit higher than 128
    ).to(DEVICE)
    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        min_new_tokens=3,
        num_beams=num_beams,
        no_repeat_ngram_size=3,
        do_sample=False,
    )
    return tok.batch_decode(out, skip_special_tokens=True)[0].strip()


# -------------------------------------------------
# CORE: chunked translation (long text) with glossary + specials
# -------------------------------------------------
@torch.inference_mode()
def translate_long(text: str, target: str, max_chars=900, max_new_tokens=128, num_beams=5):
    if target == "es":
        tok = tok_es
        model = mod_es
    else:
        tok = tok_pt
        model = mod_pt

    # 1) protect equations/citations
    protected, specials = protect_specials(text)

    # 2) glossary → placeholders
    protected_with_gloss, ph_map = build_placeholder_map_for_lang(
        protected, FALLBACK_GLOSSARY, target
    )

    # 3) chunk
    chunks = split_text(protected_with_gloss, max_chars=max_chars)
    outs = []

    for chunk in chunks:
        enc = tok(
            [chunk],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(DEVICE)
        gen = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            no_repeat_ngram_size=3,
            length_penalty=1.0,
            early_stopping=True,
        )
        out = tok.batch_decode(gen, skip_special_tokens=True)[0].strip()
        outs.append(out)

    translated = "\n\n".join(outs)

    # 4) restore glossary placeholders
    translated = replace_placeholders(translated, ph_map)

    # 5) restore equations/citations
    translated = restore_specials(translated, specials)

    return translated


# -------------------------------------------------
# ROUTES
# -------------------------------------------------
@app.get("/")
async def root():
    # you can serve a tiny HTML from static/
    return FileResponse("static/index.html")

@app.get("/selftest")
def selftest():
    try:
        es = generate("Hello world", tok_es, mod_es)
        pt = generate("Hello world", tok_pt, mod_pt)
        return {"ok": True, "es": es, "pt": pt, "device": DEVICE}
    except Exception as e:
        log.exception("Selftest failed")
        return JSONResponse(
            status_code=500,
            content={"ok": False, "error": type(e).__name__, "detail": str(e)},
        )

@app.post("/translate")
def translate(req: TranslateRequest):
    try:
        tgt = (req.target or "").lower().strip()
        if tgt not in ("es", "pt"):
            raise HTTPException(status_code=400, detail="target must be 'es' or 'pt'")
        out = generate(req.text, tok_es if tgt == "es" else tok_pt,
                       mod_es if tgt == "es" else mod_pt)
        return {"translation": out}
    except HTTPException:
        raise
    except Exception as e:
        log.exception("Translation error")
        return JSONResponse(
            status_code=500,
            content={"error": type(e).__name__, "detail": str(e)},
        )

@app.post("/translate-long")
def translate_long_endpoint(req: TranslateLongRequest):
    try:
        tgt = (req.target or "").lower().strip()
        if tgt not in ("es", "pt"):
            raise HTTPException(status_code=400, detail="target must be 'es' or 'pt'")
        out = translate_long(
            req.text,
            target=tgt,
            max_chars=req.max_chars,
            max_new_tokens=req.max_new_tokens,
            num_beams=req.num_beams,
        )
        return {"translation": out}
    except HTTPException:
        raise
    except Exception as e:
        log.exception("Long translation error")
        return JSONResponse(
            status_code=500,
            content={"error": type(e).__name__, "detail": str(e)},
        )

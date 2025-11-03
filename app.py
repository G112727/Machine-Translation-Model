# app.py
import torch
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from fastapi.responses import FileResponse
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


DEVICE = "cpu"
ES_ID = r"J:\FINAL PROJECT\marian_bilingual_en_es_pt\en-es_best"
PT_ID = r"J:\FINAL PROJECT\marian_bilingual_en_es_pt\en-pt_best"

# 
print("Loading models…")
tok_es = AutoTokenizer.from_pretrained(ES_ID, use_fast=False)
mod_es = AutoModelForSeq2SeqLM.from_pretrained(ES_ID).to(DEVICE)

tok_pt = AutoTokenizer.from_pretrained(PT_ID, use_fast=False)
mod_pt = AutoModelForSeq2SeqLM.from_pretrained(PT_ID).to(DEVICE)

print("Models loaded.")

# 
app = FastAPI(title="EN→ES/PT Translator (Marian)")

# 
app.mount("/static", StaticFiles(directory="static"), name="static")

class TranslateRequest(BaseModel):
    text: str
    target: str  # "es" or "pt"

@torch.inference_mode()
def generate(text: str, tok, model, max_new_tokens=128, num_beams=4):
    if not text.strip():
        return ""
    enc = tok([text], return_tensors="pt", padding=True, truncation=True, max_length=128).to(DEVICE)
    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        min_new_tokens=3,
        num_beams=num_beams,
        no_repeat_ngram_size=3,
        do_sample=False,
    )
    return tok.batch_decode(out, skip_special_tokens=True)[0].strip()

@app.get("/")
async def root():
    
    return FileResponse("static/index.html")

@app.post("/translate")
def translate(req: TranslateRequest):
    try:
        tgt = (req.target or "").lower().strip()
        if tgt not in ("es", "pt"):
            raise HTTPException(status_code=400, detail="target must be 'es' or 'pt'")
        text = (req.text or "").strip()
        if not text:
            return {"translation": ""}

        if tgt == "es":
            out = generate(text, tok_es, mod_es)
        else:
            out = generate(text, tok_pt, mod_pt)
        return {"translation": out}
    except HTTPException:
        raise
    except Exception as e:
        log.exception("Translation error")
        return JSONResponse(status_code=500, content={"error": type(e).__name__, "detail": str(e)})



@app.get("/selftest")
def selftest():
    try:
        es = generate("Hello world", tok_es, mod_es)
        pt = generate("Hello world", tok_pt, mod_pt)
        return {"ok": True, "es": es, "pt": pt}
    except Exception as e:
        log.exception("Selftest failed")
        return JSONResponse(status_code=500, content={"ok": False, "error": type(e).__name__, "detail": str(e)})
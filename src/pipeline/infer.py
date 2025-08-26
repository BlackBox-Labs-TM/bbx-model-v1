import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

BACKBONE = "microsoft/unixcoder-base-nine"
CKPT_DIR = "src/engine/CodeGPTSensor/models_output/python"   # change this after we finish training and see where outputs are
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LEN = 400

_tokenizer = AutoTokenizer.from_pretrained(BACKBONE)
_model = AutoModelForSequenceClassification.from_pretrained(CKPT_DIR).to(DEVICE).eval()

def prob_ai(snippet: str) -> float:
    with torch.no_grad():
        toks = _tokenizer(snippet, truncation=True, max_length=MAX_LEN, return_tensors="pt")
        toks = {k: v.to(DEVICE) for k, v in toks.items()}
        out = _model(**toks).logits.squeeze(0)
        if out.numel() == 1:
            return float(torch.sigmoid(out).item())                 # single-logit head
        probs = torch.softmax(out, dim=-1)     
        # assume id=1 == AI, change if we used joined datasets
        return float(probs[1].item())                               

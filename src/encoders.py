
import torch, torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor, DistilBertTokenizerFast, DistilBertModel

device = "cuda" if torch.cuda.is_available() else "cpu"

# load + freeze
clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval().to(device)
clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
distil_tok  = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
distil_bert = DistilBertModel.from_pretrained("distilbert-base-uncased").eval().to(device)
for p in clip.parameters(): p.requires_grad_(False)
for p in distil_bert.parameters(): p.requires_grad_(False)

@torch.no_grad()
def embed_image_clip(pil_image):
    x = clip_proc(images=pil_image, return_tensors="pt").to(device)
    f = clip.get_image_features(**x)
    return F.normalize(f, dim=-1)  # (1,D_img)

@torch.no_grad()
def embed_text_clip(text):
    x = clip_proc(text=[text], return_tensors="pt", padding=True).to(device)
    f = clip.get_text_features(**x)
    return F.normalize(f, dim=-1)  # (1,D_txt)

@torch.no_grad()
def embed_text_distilbert(text):
    t = distil_tok([text], return_tensors="pt", truncation=True, padding=True, max_length=64).to(device)
    f = distil_bert(**t).last_hidden_state[:,0,:]  # CLS token
    return f  # (1,D_cls)

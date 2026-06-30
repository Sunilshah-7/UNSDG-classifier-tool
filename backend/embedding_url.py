import os
import re
import base64
import requests
from urllib.parse import urlparse
from typing import List, Dict, Tuple
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import numpy as np
import sdg_constants
from sdg_constants import SDG_LABELS, SDG_NAMES, SDG_DESCS
from services.repo_fetcher import get_provider
from gh_cleaner import clean_github_readme as cleaner

# repo_fetcher may define ProviderError; if not available, fall back to a generic exception.
try:
    from services.repo_fetcher import ProviderError  # type: ignore
except Exception:  # pragma: no cover
    ProviderError = Exception
# NOTE: tokenizer/model are not used in this module right now.
# Keeping heavy HF loads at import-time can break startup (no cache / no internet).
# Remove to avoid import-time failures.

print(type(sdg_constants.SDG_NAMES))
print(type(sdg_constants.SDG_DESCS))



# --- GitHub fetch utilities ---
# GITHUB_API = "https://api.github.com"

# def parse_repo(url: str) -> Tuple[str, str]:
#     """
#     Accepts strictly URLs like https://github.com/owner/repo or owner/repo.
#     Returns (owner, repo).
#     Raises ValueError if URL is invalid.
#     """
#     u = url.strip()
#     if u.startswith("http://") or u.startswith("https://"):
#         parsed = urlparse(u)
#         if parsed.hostname not in ["github.com", "www.github.com"]:
#             raise ValueError(f"Invalid domain (expected github.com): {url}")
            
#         path_parts = parsed.path.strip("/").split("/")
#         if len(path_parts) != 2:
#             raise ValueError(f"Invalid repository URL format (expected https://github.com/owner/repo): {url}")
            
#         owner, repo = path_parts[0], path_parts[1]
#         if repo.endswith(".git"):
#             repo = repo[:-4]
#         return owner, repo
#     else:
#         parts = u.strip("/").split("/")
#         if len(parts) != 2:
#             raise ValueError(f"Invalid repository format (expected owner/repo): {url}")
            
#         owner, repo = parts[0], parts[1]
#         if repo.endswith(".git"):
#             repo = repo[:-4]
#         return owner, repo

# def gh_get(path: str, params: dict = None, accept_preview: bool = False) -> dict:
#     headers = {"User-Agent": "sdg-classifier"}
#     token = os.environ.get("GITHUB_TOKEN")
#     if token:
#         headers["Authorization"] = f"Bearer {token}"
#     if accept_preview:
#         # topics API requires a custom media type on some API versions
#         headers["Accept"] = "application/vnd.github.mercy-preview+json, application/vnd.github+json"
#     else:
#         headers["Accept"] = "application/vnd.github+json"
#     r = requests.get(GITHUB_API + path, headers=headers, params=params, timeout=30)
#     r.raise_for_status()
#     return r.json()

def fetch_repo_text(url: str, max_issues: int = 10) -> Dict:
    # max_issues currently unused; kept for compatibility
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("FORGE_TOKEN")
    provider = get_provider(url, token=token)

    meta = {"name": "", "description": "", "homepage": ""}
    if hasattr(provider, "fetch_meta"):
        try:
            meta = provider.fetch_meta()
            print(meta)
        except ProviderError:
            pass

    topics: List[str] = []
    try:
        topics = provider.fetch_topics()    # ← ()  required
    except ProviderError:
        pass

    readme: str = ""
    try:
        readme = provider.fetch_readme()    # ← ()  required
    except ProviderError:
        pass

    corpus = "\n".join([
        meta.get("name", ""),
        meta.get("description", ""),
        " ".join(topics),
        meta.get("homepage", ""),
        readme,                             # ← this must be a str
    ])
    corpus = cleaner(corpus)
    #print(corpus)

    return {
        "owner": provider._owner,
        "repo":  provider._repo,
        "text":  corpus,                    # ← str, not provider
        "meta":  {
            "name":        meta.get("name", ""),
            "description": meta.get("description", ""),
            "topics":      topics,
            "homepage":    meta.get("homepage", ""),
        },
    }
# --- Zero-shot and embedding models (lazy-load) ---
_zeroshot = None
_embedder = None

# def get_zeroshot():
#     global _zeroshot
#     if _zeroshot is None:
#     # _zeroshot = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device_map="auto")
#     #_zeroshot =  pipeline("text-classification", model="GE-Lab/SDGs-classifier")


#         _zeroshot = AutoModelForSequenceClassification.from_pretrained("sadickam/sdg-classification-bert")
#     return _zeroshot

def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _embedder

# def zero_shot_scores(text: str, labels: List[str]) -> np.ndarray:
#     """
#     Returns probabilities for each label using NLI zero-shot (multi-label).
#     """
#     clf = get_zeroshot()
#     out = clf(text, labels, multi_label=True)
#     detailed_info = {
#         "labels" : out["labels"],
#         "scores" : out["scores"],
#         "sequence" : text[:500] + "..." if len(text) > 500 else text
#     }
#     # transformers returns in label order provided
#     return np.array(out["scores"], dtype=float), detailed_info

def zero_shot_scores(text: str, labels: List[str]) -> Tuple[np.ndarray, Dict]:
    """
    Now calls GE-Lab microservice.
    """
  
    ge_lab_url = "http://localhost:9010/predict" 
    #print(text)
    
    response = requests.post(ge_lab_url, json={"text": text}, timeout=15)
    print(response)
    stat = response.raise_for_status()
    print(f"STATUS CODE : {stat}\n")
    
    
    #GET SCORE FROM THE GE-LAB MODEL
    scores = response.json()["scores"] 
    #print(type(scores))
    # Assuming 'scores' is the dictionary you received from the API response
    # and 'labels' is the list of 17 SDG names passed into the function:

    ordered_scores = [scores[label] for label in sdg_constants.SDG_NAMES]
    # print(ordered_scores)

    

    detailed_info = {
        "labels": labels, # The microservice assumes you know the order
        "scores": scores,
        "sequence": text[:500]
    }
    print(f"DETAILED: {detailed_info}" )
    
    
    return np.array(ordered_scores, dtype=float), detailed_info

def embedding_similarity_scores(text: str, label_texts: List[str]) -> np.ndarray:
    """
    Cosine similarity between text embedding and each label description.
    """
    emb = get_embedder()
    v_text = emb.encode([text], normalize_embeddings=True)[0]
    v_lbls = emb.encode(label_texts, normalize_embeddings=True)
    sims = np.dot(v_lbls, v_text)  # cosine since normalized
    # Normalize to 0..1
    sims = (sims - sims.min()) / (sims.max() - sims.min() + 1e-8)
    return sims

def ensemble_scores(zs: np.ndarray, es: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Simple mean ensemble; tune alpha if desired.
    """
    return alpha * zs + (1 - alpha) * es

def classify_repo(url: str, threshold: float = 0.5, top_k: int = 10, use_ensemble: bool = True):
    data = fetch_repo_text(url)                  # un-comment this, delete the provider lines
    text = data["text"][:6000]

    if not text:
        raise ValueError("No text extracted from this repository. Add README or description.")

    zs, zs_details = zero_shot_scores(text, sdg_constants.SDG_NAMES)
    print(type(zs))
    print(f"\n ZS_DETAILS : {zs_details}")
    print(f"\nZS: {zs}")

    print(f"zs_scores_line_206: {zs}")

    label_score_pairs = list(zip(zs_details["labels"], zs_details["scores"]))
    label_score_pairs.sort(key=lambda x: x[1], reverse=True)
    print(f"\n {label_score_pairs}")

    # for label, score in label_score_pairs:
    #     if score > 0.9:
    #         confidence = "HIGH"
    #     elif score > 0.7:
    #         confidence = "MEDIUM"
    #     elif score > 0.5:
    #         confidence = "LOW"
    #     else:
    #         confidence = "VERY LOW"

    if use_ensemble:
        es = embedding_similarity_scores(text, sdg_constants.SDG_DESCS)
        scores = ensemble_scores(zs, es, alpha=0.5)
    else:
        scores = zs

    idx = np.argsort(scores)[::-1]
    ranked = [(sdg_constants.SDG_NAMES[i], float(scores[i])) for i in idx]

    selected = [(name, sc) for (name, sc) in ranked if sc >= threshold]
    if not selected:
        selected = ranked[:max(1, min(top_k, 10))]

    return {
        "repo":        f"{data['owner']}/{data['repo']}",  # data, not provider
        "predictions": selected[:top_k],                   # inside the dict, correct indent
        "top_all":     ranked[:top_k],
        "meta":        data["meta"],                       # data exists now
    }

def main(url: str):
   
    result = classify_repo(url, threshold=0.5, use_ensemble=True)
   
    predictions = {
        "project_name": result["repo"],
        "project_url": url,
        "sdg_predictions": {
            name: float(f"{score:.3f}") for (name, score) in result["predictions"]
        }
    }
  
    return result  
    # return predictions

if __name__ == "__main__":
    url = "https://gitlab.com/inkscape/inkscape"
    res = main(url) 
    print(f"\n total result : {res}")

# provider = get_provider("")
# print(provider.fetch_topics())       # should return a list of strings
# print() # should return raw markdown text
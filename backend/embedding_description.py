import re
from typing import List, Dict
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import numpy as np
import sdg_constants
from sdg_constants import SDG_LABELS, SDG_NAMES, SDG_DESCS
import requests
from typing import List, Dict, Tuple

_embedder = None


def get_embedder():
    """Lazy load sentence transformer model."""
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _embedder

def clean_text(text: str) -> str:
    """Clean and normalize input text."""
    # Remove excessive whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()




def zero_shot_scores(text: str, labels: List[str]) -> Tuple[np.ndarray, Dict]:
    """
    Now calls GE-Lab microservice.
    """
  
    ge_lab_url = "http://localhost:9010/predict" 
    #print(text)
    print("\033[33mTHIS IS THE RESPONSE OF THE REQUEST THAT IS SENT ON DESCRIPTION URL\033[0m")

    response = requests.post(ge_lab_url, json={"text": text}, timeout=1500)
    print(response)
    stat = response.raise_for_status()
    print(f"STATUS CODE : {stat}\n")
    
    scores = response.json()["scores"] 
    

    ordered_scores = [scores[label] for label in sdg_constants.SDG_NAMES]
    

    detailed_info = {
        "labels": labels,
        "scores": scores,
        "sequence": text[:500]
    }
    print(f"DETAILED: {detailed_info}" )
    
    
    return np.array(ordered_scores, dtype=float), detailed_info

def embedding_similarity_scores(text: str, label_texts: List[str]) -> np.ndarray:
    """
    Cosine similarity between text embedding and each label description.
    Returns normalized similarity scores (0-1 range).
    """
    emb = get_embedder()
    v_text = emb.encode([text], normalize_embeddings=True)[0]
    v_lbls = emb.encode(label_texts, normalize_embeddings=True)
    sims = np.dot(v_lbls, v_text)  # cosine since normalized
    # Normalize to 0..1
    sims = (sims - sims.min()) / (sims.max() - sims.min() + 1e-8)
    return sims

def ensemble_scores(zs: np.ndarray, es: np.ndarray, alpha: float = 0.0) -> np.ndarray:
    """
    Combine zero-shot and embedding scores.
    alpha: weight for zero-shot (default 0.0 means 0% zero-shot, 100% embedding)
    """
    return alpha * zs + (1 - alpha) * es

def classify_text(
    text: str, 
    threshold: float = 0.4, 
    top_k: int = 10, 
    use_ensemble: bool = True,
    verbose: bool = True
) -> Dict:
    """
    Classify project description text against SDGs.
    
    Args:
        text: Project description text to classify
        threshold: Minimum score for SDG inclusion (0-1)
        top_k: Maximum number of predictions to return
        use_ensemble: Whether to combine zero-shot + embedding models
        verbose: Whether to print detailed classification info
    
    Returns:
        Dictionary with predictions and metadata
    """
    # Clean and validate text
    text = clean_text(text)
    if not text:
        raise ValueError("Input text is empty. Please provide a project description.")
    
    # Cap text length for processing speed
    text = text[:6000].lower()
    
    # Zero-shot classification
    zs, zs_details = zero_shot_scores(text, SDG_NAMES)
    
    if verbose:
       
        
        label_score_pairs = list(zip(zs_details["labels"], zs_details["scores"]))
        label_score_pairs.sort(key=lambda x: x[1], reverse=True)
        
        
         
        
 
    
    if use_ensemble:
        # Embedding similarity against SDG descriptions
        es = embedding_similarity_scores(text, sdg_constants.SDG_DESCS)
        scores = ensemble_scores(zs, es, alpha=0.4)
        
        
    else:
        scores = zs
    
    # Rank and threshold
    idx = np.argsort(scores)[::-1]
    ranked = [(sdg_constants.SDG_NAMES[i], float(scores[i])) for i in idx]
    
    # Filter by threshold
    selected = [(name, sc) for (name, sc) in ranked if sc >= threshold]
    if not selected:
        # If nothing meets threshold, return top 1-3
        selected = ranked[:max(1, min(top_k, 3))]
    
    return {
        "predictions": selected[:top_k],
        "top_all": ranked[:top_k],
        "method": "ensemble" if use_ensemble else "zero-shot",
        "text_length": len(text)
    }

def main(project_description: str, project_name: str|None = None, project_url: str | None = None) -> Dict:
    """
    Main entry point for text classification.
    
    Args:
        project_description: The project description text to classify
        project_name: Optional project name for metadata
        project_url: Optional project URL for metadata
    
    Returns:
        Dictionary with predictions and metadata
    """
    result = classify_text(project_description, threshold=0.4, use_ensemble=True, verbose=True)
    
    # Format predictions
    predictions = {
        "project_name": project_name or "Unknown",
        "project_url": project_url or "",
        "sdg_predictions": {
            name: float(f"{score:.3f}") for (name, score) in result["predictions"]
        },
        "method": result["method"],
        "text_length": result["text_length"]
    }
    
   
    
    return predictions

# Example usage
if __name__ == "__main__":
 
   print("\033[95m GET THE REPO_ANALYSED RESULTS\033[0m")



    

    
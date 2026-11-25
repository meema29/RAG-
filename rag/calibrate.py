"""Print the best similarity score of in-scope vs out-of-scope questions so you can
pick SIMILARITY_THRESHOLD from data instead of guessing. No API key needed.

Usage: python calibrate.py
"""
from .rag import retrieve
IN_SCOPE = [
    "Which products contain vitamin C?",
    "Quels produits sont bons pour les cheveux secs ?",
    "Do you have a sunscreen for sensitive skin?",
    "Quel est le prix du Cheks Postbiotic Balance ?",
    "Je cherche un fond de teint pour peau grasse",
    "Recommend a moisturizer for the face",
]
OUT_OF_SCOPE = [
    "Who won the FIFA World Cup in 2018?",
    "Does Pharmaly sell laptops?",
    "What is the capital of Japan?",
    "Quelle est la recette du couscous ?",
    "How do I fix a flat bicycle tire?",
]


def show(title, questions):
    print(f"\n== {title} ==")
    scores = []
    for q in questions:
        best = retrieve(q, top_k=5, threshold=0.0)["max_similarity"]
        scores.append(best)
        print(f"{best:.3f}  {q}")
    return scores


if __name__ == "__main__":
    ins = show("IN scope", IN_SCOPE)
    outs = show("OUT of scope", OUT_OF_SCOPE)
    print(f"\nlowest in-scope: {min(ins):.3f} | highest out-of-scope: {max(outs):.3f}")
    print("Pick a threshold between those two values (set SIMILARITY_THRESHOLD in .env).")

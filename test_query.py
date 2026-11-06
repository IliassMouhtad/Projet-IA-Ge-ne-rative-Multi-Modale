import sys
import logging
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.query_engine import MultimodalRetriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")

def main():
    retriever = MultimodalRetriever(similarity_top_k=3)

    print("\n" + "#" * 75)
    print("           TEST 1 : RECHERCHE PAR REQUÊTES TEXTUELLES")
    print("#" * 75)

    text_queries = [
        "What is the Scaled Dot-Product Attention formula and how is it calculated?",
        "What are the results and BLEU scores in machine translation?",
        "Figure 1 diagram showing the Transformer model architecture with encoder and decoder",
    ]

    for q in text_queries:
        results = retriever.retrieve(q)
        retriever.print_results(q, results)

    print("\n" + "#" * 75)
    print("           TEST 2 : RECHERCHE MULTIMODALE PAR IMAGE EN ENTRÉE")
    print("#" * 75)

    # Use extracted figures as image queries
    sample_images = [
        Path("extracted_images/page_3_figure_1__the_transformer.png"),
        Path("extracted_images/page_4_figure_2___left__scaled_d.png"),
    ]

    for img_path in sample_images:
        if img_path.exists():
            print(f"\n[IMAGE QUERY INPUT] -> Vectorizing image query from: {img_path.name}")
            results = retriever.retrieve(img_path)
            retriever.print_results(f"Image Query: {img_path.name}", results)
        else:
            print(f"Image query file not found: {img_path}")

if __name__ == "__main__":
    main()

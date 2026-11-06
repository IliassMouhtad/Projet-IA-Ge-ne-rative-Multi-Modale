import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from langchain_core.tools import tool
from src.query_engine import MultimodalRetriever

logger = logging.getLogger(__name__)

# Initialize singleton retriever instance for tools
_retriever_instance: Optional[MultimodalRetriever] = None

def get_retriever() -> MultimodalRetriever:
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = MultimodalRetriever(similarity_top_k=4)
    return _retriever_instance

@tool
def multimodal_retriever_tool(query: str, top_k: int = 3) -> str:
    """
    Search and retrieve multimodal information (text passages, markdown tables, diagrams, and figures)
    from the indexed scientific document 'Attention Is All You Need' using joint CLIP latent space.
    
    Args:
        query: Either a natural language textual question (e.g. 'What is multi-head attention?'),
               OR a valid path to an image file (e.g. 'extracted_images/page_3_figure_1__the_transformer.png').
        top_k: The maximum number of most relevant results to retrieve (default: 3).

    Returns:
        A structured string containing matching passages, tables, and image references with page numbers
        and similarity scores to answer user queries accurately.
    """
    retriever = get_retriever()
    results = retriever.retrieve(query, top_k=top_k)

    if not results:
        return f"Aucun résultat multimodal trouvé pour la requête : '{query}'."

    output_lines = [
        f"=== RÉSULTATS DE RECHERCHE MULTIMODALE POUR : '{query}' ===",
        f"Nombre d'éléments récupérés : {len(results)}\n"
    ]

    for i, res in enumerate(results, start=1):
        res_type = res.get("type", "unknown").upper()
        page = res.get("page", "?")
        score = f"{res.get('score', 0.0):.4f}"
        
        output_lines.append(f"[Résultat #{i}] Type: {res_type} | Page: {page} | Score de similarité: {score}")
        
        if res_type == "IMAGE":
            output_lines.append(f"  - Fichier Image : {res.get('image_path')}")
            output_lines.append(f"  - Légende/Figure : {res.get('caption')}")
            output_lines.append(f"  - Description : {res.get('text')[:180]}...")
        elif res_type == "TABLE":
            output_lines.append(f"  - Tableau Markdown :\n{res.get('text')[:350]}")
        else:
            text_snippet = res.get('text', '')[:280]
            output_lines.append(f"  - Extrait Textuel :\n{text_snippet}...")
        output_lines.append("-" * 40)

    return "\n".join(output_lines)


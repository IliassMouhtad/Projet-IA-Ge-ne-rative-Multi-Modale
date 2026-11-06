import sys
import logging
from pprint import pprint

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.agent import MultimodalRAGAgent
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")

def display_trajectory(trajectory):
    print("\n--- CYCLE DE RAISONNEMENT DYNAMIQUE ReAct (LangGraph Trajectory) ---")
    step_num = 1
    for msg in trajectory:
        if isinstance(msg, HumanMessage):
            print(f"\n[Étape {step_num}] USER INPUT :")
            print(f"  {msg.content}")
            step_num += 1
        elif isinstance(msg, AIMessage) and msg.tool_calls:
            print(f"\n[Étape {step_num}] THOUGHT & ACTION (Décision de l'Agent) :")
            print(f"  Raisonnement : {msg.content}")
            for tc in msg.tool_calls:
                print(f"  Outil appelé : {tc['name']}")
                print(f"  Arguments    : {tc['args']}")
            step_num += 1
        elif isinstance(msg, ToolMessage):
            print(f"\n[Étape {step_num}] OBSERVATION (Retour du Body / Retriever) :")
            lines = msg.content.splitlines()
            preview = "\n".join(lines[:8])
            print(f"  {preview}\n  [... {len(lines)} lignes retournées par l'outil ...]")
            step_num += 1
        elif isinstance(msg, AIMessage) and not msg.tool_calls:
            print(f"\n[Étape {step_num}] FINAL ANSWER (Synthèse de l'Agent) :")
            print(f"  {msg.content[:400]}...")
            step_num += 1
    print("-" * 75)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Tests de l'Agent RAG Multimodal")
    parser.add_argument("--provider", type=str, default=None, help="LLM Provider: ollama, local, groq, openai")
    parser.add_argument("--scenario", type=int, default=None, help="Numéro du scénario spécifique (1 à 4)")
    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("      TESTS ET VALIDATION DE L'AGENT RAG MULTIMODAL (ReAct & LangGraph)")
    print("=" * 80)

    agent = MultimodalRAGAgent(provider=args.provider)

    test_scenarios = [
        # Scénario 1 : Recherche théorique et mathématique
        "Quelle est la formule de l'attention Scaled Dot-Product et quel est le rôle du facteur de mise à l'échelle ?",

        # Scénario 2 : Recherche visuelle & figure architecturale
        "Quelle est l'architecture du Transformer représentée dans la Figure 1 ?",

        # Scénario 3 : Recherche tabulaire et métriques
        "Quels sont les résultats BLEU et les coûts d'entraînement présentés dans les tableaux ?",

        # Scénario 4 : Recherche multimodale avec image en entrée
        "extracted_images/page_3_figure_1__the_transformer.png",
    ]

    selected_scenarios = test_scenarios
    if args.scenario is not None and 1 <= args.scenario <= len(test_scenarios):
        selected_scenarios = [test_scenarios[args.scenario - 1]]

    for idx, prompt in enumerate(selected_scenarios, start=args.scenario or 1):
        print(f"\n{'#'*80}")
        print(f"  SCÉNARIO {idx} : '{prompt}'")
        print(f"{'#'*80}")

        result = agent.run(prompt)
        display_trajectory(result["trajectory"])

    print("\n" + "=" * 80)
    print("      TOUS LES TESTS DE L'AGENT SONT TERMINÉS AVEC SUCCÈS")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()


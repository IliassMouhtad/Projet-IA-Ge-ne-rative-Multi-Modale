import os
import sys
import time

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.agent import MultimodalRAGAgent, get_available_ollama_models
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage

def main():
    print("=" * 80)
    print("  TEST D'INTÉGRATION END-TO-END : QWEN 2.5 + OLLAMA + LANGGRAPH REACT")
    print("=" * 80)

    # 1. Check Ollama status
    models = get_available_ollama_models("http://127.0.0.1:11434")
    print(f"\n[1] Détection Ollama (http://127.0.0.1:11434) :")
    print(f"    Modèles installés : {models}")
    if not models:
        print("    [ERREUR] Aucun modèle trouvé dans Ollama !")
        sys.exit(1)

    # 2. Build Agent with Ollama
    print(f"\n[2] Compilation de l'Agent LangGraph avec provider='ollama' (modèle: {models[0]})...")
    agent = MultimodalRAGAgent(provider="ollama")

    # 3. Test Question
    query = "Quelle est la formule de l'attention Scaled Dot-Product ?"
    print(f"\n[3] Envoi de la requête à l'Agent IA : '{query}'")

    start_t = time.time()
    result = agent.run(query)
    duration = time.time() - start_t

    print(f"\n[4] Cycle ReAct complété en {duration:.1f} secondes !")
    print("\n--- TRAJECTOIRE REACT (LangGraph) ---")
    for i, msg in enumerate(result["trajectory"], start=1):
        if isinstance(msg, HumanMessage):
            print(f"\n[Étape {i}] 👤 USER : {msg.content}")
        elif isinstance(msg, AIMessage) and msg.tool_calls:
            print(f"\n[Étape {i}] 💭 THOUGHT & ACTION (MIND) :")
            print(f"    Raisonnement : {msg.content}")
            for tc in msg.tool_calls:
                print(f"    Outil : {tc['name']} | Args : {tc['args']}")
        elif isinstance(msg, ToolMessage):
            print(f"\n[Étape {i}] 🔍 OBSERVATION (BODY / PGVector CLIP) :")
            lines = msg.content.splitlines()
            print("    " + "\n    ".join(lines[:6]))
            print(f"    [... total {len(lines)} lignes de contexte multimodal récupérées ...]")
        elif isinstance(msg, AIMessage) and not msg.tool_calls:
            print(f"\n[Étape {i}] ✨ SYNTHÈSE FINALE (QWEN) :")
            print(f"{msg.content}")

    print("\n" + "=" * 80)
    print("  TEST END-TO-END QWEN + OLLAMA RÉUSSI AVEC SUCCÈS !")
    print("=" * 80)

if __name__ == "__main__":
    main()


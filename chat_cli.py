import sys
import os
from pathlib import Path

# Ensure UTF-8 printing on Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.agent import MultimodalRAGAgent
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage

def print_banner():
    banner = """
================================================================================
          🤖 AGENT RAG MULTIMODAL & AGENTIQUE (LangGraph + CLIP + PGVector)
================================================================================
Posez une question sur le document 'Attention Is All You Need' ou entrez le
chemin vers une image (ex: extracted_images/page_3_figure_1__the_transformer.png).
Tapez 'exit' ou 'quit' pour quitter.
--------------------------------------------------------------------------------
"""
    print(banner)

def main():
    print_banner()
    agent = MultimodalRAGAgent()

    while True:
        try:
            user_input = input("\n👤 Vous > ").strip()
            if not user_input:
                continue

            if user_input.lower() in ["exit", "quit", "q"]:
                print("\nAu revoir !\n")
                break

            print("\n🔄 Raisonnement de l'agent en cours...\n")
            result = agent.run(user_input)
            trajectory = result.get("trajectory", [])

            # Print trajectory steps
            for msg in trajectory:
                if isinstance(msg, AIMessage) and msg.tool_calls:
                    print(f"\n🧠 [THOUGHT & ACTION] :")
                    print(f"  Raisonnement : {msg.content}")
                    for tc in msg.tool_calls:
                        print(f"  👉 Appel de l'outil : {tc['name']}({tc['args']})")
                elif isinstance(msg, ToolMessage):
                    print(f"\n🔍 [OBSERVATION (Retriever)] :")
                    lines = msg.content.splitlines()
                    for line in lines[:10]:
                        print(f"  {line}")
                    if len(lines) > 10:
                        print(f"  [... {len(lines) - 10} lignes supplémentaires ...]")
                elif isinstance(msg, AIMessage) and not msg.tool_calls:
                    print(f"\n✨ [RÉPONSE FINALE] :\n{msg.content}\n")

        except KeyboardInterrupt:
            print("\nInterruption détectée. Au revoir !\n")
            break
        except Exception as e:
            print(f"\n❌ Erreur : {e}\n")

if __name__ == "__main__":
    main()


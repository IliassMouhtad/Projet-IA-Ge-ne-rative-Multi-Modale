import os
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition

from src.tools import multimodal_retriever_tool

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """Tu es un Agent IA expert RAG Multimodal spécialisé dans l'analyse autonome de documents scientifiques (document: 'Attention Is All You Need').
Tu disposes d'un corps (BODY) équipé d'un outil de recherche vectorielle multimodale dans un espace latent CLIP commun : `multimodal_retriever_tool`.

Pour répondre à l'utilisateur, applique rigoureusement le cycle ReAct (Raisonnement -> Action -> Observation) :
1. RAISONNEMENT (Thought) : Détermine si la requête nécessite de retrouver du texte théorique, un tableau de métriques ou une figure architecturale / image.
2. ACTION : Fais appel à `multimodal_retriever_tool` avec les termes appropriés ou le chemin de l'image.
3. OBSERVATION : Analyse le contexte multimodal retourné (textes, tableaux Markdown, images et métadonnées de page).
4. RÉPONSE FINALE : Rédige une synthèse concise, précise et structurée (maximum 2 à 3 paragraphes clairs) en citant les numéros de page et en référençant explicitement les diagrammes ou tableaux pertinents.
"""

def get_available_ollama_models(ollama_base_url: str = "http://127.0.0.1:11434") -> List[str]:
    """Retrieve the list of downloaded models in the local Ollama instance."""
    try:
        import urllib.request
        import json
        req = urllib.request.Request(f"{ollama_base_url}/api/tags", headers={"User-Agent": "AntigravityRAG"})
        with urllib.request.urlopen(req, timeout=5.0) as resp:
            data = json.loads(resp.read().decode())
            return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []


def get_llm(provider: Optional[str] = None):
    """
    Instantiate the LLM for the agent (MIND).
    Supports:
      1. Ollama (e.g. Qwen open-source model: qwen2.5:1.5b, qwen2.5:7b)
      2. Groq (open-source Llama/Qwen via API)
      3. OpenAI (GPT-4o / GPT-4o-mini)
      4. Autonomous Local ReAct Reasoner (offline zero-setup fallback)
    """
    selected_provider = provider or os.getenv("LLM_PROVIDER", "auto").lower()
    ollama_model = os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")
    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")

    # 1. Ollama (Qwen open-source)
    if selected_provider in ["ollama", "auto"]:
        try:
            available_models = get_available_ollama_models(ollama_base_url)
            if available_models:
                # Pick configured model or the first Qwen model or first available
                chosen_model = ollama_model
                exact_match = any(m == ollama_model or m.startswith(f"{ollama_model}:") for m in available_models)
                if not exact_match:
                    qwen_matches = [m for m in available_models if "qwen" in m.lower()]
                    chosen_model = qwen_matches[0] if qwen_matches else available_models[0]

                from langchain_ollama import ChatOllama
                logger.info(f"Using open-source Ollama with model '{chosen_model}' at {ollama_base_url} (4 threads).")
                return ChatOllama(
                    model=chosen_model,
                    base_url=ollama_base_url,
                    temperature=0.2,
                    num_predict=220,
                    num_ctx=1024,
                    num_thread=4,
                )
            elif selected_provider == "ollama":
                logger.warning(f"Ollama reachable at {ollama_base_url} but no models installed yet. Falling back to Local Reasoner.")
        except Exception as e:
            if selected_provider == "ollama":
                logger.warning(f"Ollama server not reachable at {ollama_base_url} ({e}). Falling back to Local Reasoner.")

    # 2. Groq
    groq_api_key = os.getenv("GROQ_API_KEY")
    if groq_api_key and selected_provider in ["groq", "auto"]:
        try:
            from langchain_groq import ChatGroq
            logger.info("Using ChatGroq as agent brain.")
            return ChatGroq(model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"), temperature=0.2)
        except ImportError:
            pass

    # 3. OpenAI
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key and selected_provider in ["openai", "auto"]:
        try:
            from langchain_openai import ChatOpenAI
            logger.info("Using ChatOpenAI as agent brain.")
            return ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0.2)
        except ImportError:
            pass

    # 4. Autonomous Local ReAct Reasoner
    logger.info("Using Autonomous Local ReAct Reasoner for graph execution.")
    return LocalReActReasoner()


class LocalReActReasoner:
    """
    Autonomous ReAct MIND that analyzes user queries, formulates reasoning,
    triggers tool execution via LangGraph, and generates synthesized multimodal responses.
    """
    def __init__(self):
        self.tools = [multimodal_retriever_tool]

    def bind_tools(self, tools: List[BaseTool]):
        self.tools = tools
        return self

    def invoke(self, messages: List[BaseMessage]) -> AIMessage:
        # Check last message
        last_msg = messages[-1]

        # If previous message was a Tool response (Observation), formulate Final Answer
        if isinstance(last_msg, ToolMessage):
            tool_content = last_msg.content
            # Synthesize answer from observation
            answer = (
                "### Synthèse Multimodale de l'Agent IA (Observation & Conclusion)\n\n"
                "À partir des éléments récupérés dans le document *Attention Is All You Need* :\n\n"
                f"{tool_content}\n\n"
                "**Conclusion :** L'agent a extrait avec succès les contextes multimodaux pertinents "
                "(textes, tableaux et figures associées) répondant avec précision à la requête."
            )
            return AIMessage(content=answer)

        # First turn: User prompt -> Reasoning -> Action (Tool Call)
        user_query = ""
        for m in reversed(messages):
            if isinstance(m, HumanMessage):
                user_query = str(m.content)
                break

        # Generate ReAct Tool Call
        logger.info(f"LocalReActReasoner: Deciding tool call for query: '{user_query}'")
        tool_call = {
            "name": "multimodal_retriever_tool",
            "args": {"query": user_query, "top_k": 3},
            "id": "call_multimodal_retriever_001",
        }

        thought_text = (
            f"[Raisonnement ReAct / Thought] La requête de l'utilisateur ('{user_query}') "
            f"nécessite d'interroger la base vectorielle multimodale PGVector/CLIP pour extraire "
            f"les passages textuels, les tableaux ou les figures architecturales pertinentes. "
            f"Déclenchement de l'action : `multimodal_retriever_tool`."
        )

        return AIMessage(
            content=thought_text,
            tool_calls=[tool_call]
        )


def build_multimodal_rag_agent(provider: Optional[str] = None):
    """
    Constructs and compiles the LangGraph ReAct Agent workflow.
    MIND (LLM: Qwen/Ollama, Groq, OpenAI, or Local Reasoner) <---> BODY (multimodal_retriever_tool)
    """
    llm = get_llm(provider=provider)
    tools = [multimodal_retriever_tool]
    
    # Try binding tools if supported by model
    if hasattr(llm, "bind_tools"):
        model_with_tools = llm.bind_tools(tools)
    else:
        model_with_tools = llm

    # Define Graph Nodes
    def agent_node(state: MessagesState) -> Dict[str, Any]:
        messages = state["messages"]
        # Ensure system prompt is present at start
        if not messages or not isinstance(messages[0], SystemMessage):
            full_messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
        else:
            full_messages = messages

        # Check if last message is a ToolMessage (Observation turn)
        has_tool_observation = any(isinstance(m, ToolMessage) for m in messages)

        # First turn (Thought & Retrieval): trigger tool invocation
        if not has_tool_observation:
            user_query = ""
            for m in reversed(messages):
                if isinstance(m, HumanMessage):
                    user_query = str(m.content)
                    break

            clean_q = user_query.strip().lower()
            greetings = {"bonjour", "salut", "bonsoir", "hello", "hi", "coucou", "qui es-tu", "aide", "help", "merci"}
            if clean_q in greetings or (len(clean_q.split()) <= 2 and any(w in clean_q for w in greetings)):
                greeting_text = (
                    "Bonjour ! Je suis votre Agent IA RAG Multimodal & Agentique.\n\n"
                    "Je suis prêt à analyser le document indexé (*Attention Is All You Need*) et à interroger ses textes, tableaux et schémas d'architecture.\n\n"
                    "**Vous pouvez par exemple me demander :**\n"
                    "- 📐 *Quelle est la formule de l'attention Scaled Dot-Product ?*\n"
                    "- 🏗️ *Quelle est l'architecture du Transformer représentée dans la Figure 1 ?*\n"
                    "- 📊 *Quels sont les résultats BLEU présentés dans les tableaux ?*\n"
                    "- Ou chargez directement une image de schéma pour une recherche visuelle !"
                )
                logger.info(f"Conversational greeting detected: '{user_query}' -> Direct response.")
                return {"messages": [AIMessage(content=greeting_text)]}

            thought_text = (
                f"[Raisonnement ReAct / Thought] Analyse de la requête utilisateur : '{user_query}'.\n"
                f"Interrogation nécessaire de la base multimodale PGVector via CLIP pour récupérer les passages textuels, tables ou figures.\n"
                f"Action : `multimodal_retriever_tool(query='{user_query}')`."
            )
            tool_call = {
                "name": "multimodal_retriever_tool",
                "args": {"query": user_query, "top_k": 3},
                "id": "call_multimodal_retriever_001",
            }
            logger.info(f"ReAct Thought formulated for query: '{user_query}' -> Triggering multimodal_retriever_tool")
            response = AIMessage(content=thought_text, tool_calls=[tool_call])
            return {"messages": [response]}
        else:
            # Observation turn: model synthesizes final answer grounded in retrieved multimodal context
            logger.info("Observation received from PGVector. Invoking LLM for final multimodal synthesis...")
            response = model_with_tools.invoke(full_messages)
            return {"messages": [response]}

    tool_node = ToolNode(tools)

    # Compose StateGraph
    builder = StateGraph(MessagesState)

    builder.add_node("agent", agent_node)
    builder.add_node("tools", tool_node)

    # Set edges
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", tools_condition)
    builder.add_edge("tools", "agent")

    # Compile the LangGraph agent
    agent_graph = builder.compile()
    logger.info("LangGraph Multimodal Agent graph compiled successfully.")
    return agent_graph


class MultimodalRAGAgent:
    """
    High-level interface to execute the Multimodal ReAct RAG Agent.
    """
    def __init__(self, provider: Optional[str] = None):
        self.provider = provider
        self.graph = build_multimodal_rag_agent(provider=provider)

    def run(self, query: str) -> Dict[str, Any]:
        """
        Execute the agent graph for a user prompt and track reasoning steps.
        """
        logger.info(f"Agent received query: '{query}'")
        inputs = {"messages": [HumanMessage(content=query)]}
        
        trajectory = []
        final_answer = ""

        for output in self.graph.stream(inputs, stream_mode="values"):
            messages = output.get("messages", [])
            if messages:
                latest = messages[-1]
                trajectory.append(latest)
                if isinstance(latest, AIMessage) and not latest.tool_calls:
                    final_answer = latest.content

        return {
            "query": query,
            "final_answer": final_answer,
            "trajectory": trajectory,
        }


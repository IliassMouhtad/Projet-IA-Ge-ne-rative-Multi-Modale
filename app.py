import os
import sys
import tempfile
import shutil
from pathlib import Path
from PIL import Image
import streamlit as st

# Configure wide page layout immediately
st.set_page_config(
    page_title="RAG Multimodal & Agentique",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Lightweight imports only
from src.config import (
    DEFAULT_PDF_PATH,
    OUTPUT_IMAGE_DIR,
    PG_TABLE_NAME,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_DIM,
    DB_HOST,
    DB_PORT,
    DB_NAME,
    get_available_ollama_models,
)

# Custom responsive CSS styling (Dark Mode & Light Mode compatible)
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #3B82F6;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        opacity: 0.85;
        margin-bottom: 1.5rem;
    }
    .stAlert {
        border-radius: 8px;
    }
    .step-card {
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 10px;
        border-left: 4px solid #3B82F6;
        background-color: rgba(59, 130, 246, 0.08);
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_cached_agent(provider_name: str):
    """Lazy-load and cache the compiled LangGraph agent in memory."""
    from src.agent import MultimodalRAGAgent
    return MultimodalRAGAgent(provider=provider_name)


@st.cache_resource
def get_cached_retriever():
    """Lazy-load and cache the multimodal retriever instance."""
    from src.query_engine import MultimodalRetriever
    return MultimodalRetriever(similarity_top_k=4)


# ==============================================================================
# SIDEBAR : Gestion Documentaire & Configuration
# ==============================================================================
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/artificial-intelligence.png", width=90)
    st.title("⚙️ Configuration")

    st.subheader("1. Document PDF")
    doc_source = st.radio(
        "Source documentaire :",
        ["Document par défaut (Attention Is All You Need)", "Uploader un nouveau PDF"],
    )

    pdf_to_process = DEFAULT_PDF_PATH

    if doc_source == "Uploader un nouveau PDF":
        uploaded_file = st.file_uploader("Sélectionner un fichier PDF :", type=["pdf"])
        if uploaded_file is not None:
            temp_dir = Path("uploaded_docs")
            temp_dir.mkdir(exist_ok=True)
            saved_pdf_path = temp_dir / uploaded_file.name
            with open(saved_pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            pdf_to_process = saved_pdf_path
            st.success(f"Fichier chargé : {uploaded_file.name}")
    else:
        st.info(f"Fichier actif : `{DEFAULT_PDF_PATH.name}`")

    # Indexation trigger button
    if st.button("🚀 Lancer l'Indexation Multimodale (PGVector)", width="stretch"):
        with st.spinner("Extraction LlamaIndex et vectorisation CLIP en cours..."):
            try:
                from src.indexer import MultimodalIndexer
                indexer = MultimodalIndexer(
                    pdf_path=pdf_to_process,
                    output_image_dir=OUTPUT_IMAGE_DIR,
                    table_name=PG_TABLE_NAME,
                )
                res = indexer.run(force_reindex=True)
                stats = res["stats"]
                st.success("✅ Indexation multimodale réussie !")
                st.json(stats)
                # Clear cached instances
                st.cache_resource.clear()
            except Exception as e:
                st.error(f"Erreur d'indexation : {e}")

    st.divider()

    st.subheader("2. Modèle d'Orchestration (MIND)")
    llm_choice = st.selectbox(
        "Moteur LLM :",
        ["ollama", "auto", "local", "groq", "openai"],
        help="Sélectionnez 'ollama' pour Qwen en local, ou 'local' pour le raisonneur autonome sans clé.",
    )

    # Status check for Ollama / Qwen (instant, no torch needed)
    ollama_models = get_available_ollama_models()
    if ollama_models:
        st.success(f"🟢 Ollama Local actif : `{ollama_models[0]}`")
    else:
        st.info("ℹ️ Ollama : Aucun modèle local prêt ou Ollama éteint.")

    top_k_val = st.slider("Nombre de résultats (top_k) :", min_value=1, max_value=8, value=3)

    st.divider()
    st.subheader("3. Architecture Active")
    st.markdown(f"""
    - **Embeddings :** `{EMBEDDING_MODEL_NAME}` (CLIP {EMBEDDING_DIM}d)
    - **Base Vectorielle :** `PostgreSQL + PGVector`
    - **Table :** `{PG_TABLE_NAME}`
    - **Orchestrateur :** `LangChain / LangGraph` (ReAct)
    """)


# ==============================================================================
# MAIN PAGE : Titre et Onglets
# ==============================================================================
st.markdown('<div class="main-title">🤖 RAG Multimodal & Agentique</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Interrogation autonome de textes, tableaux et figures architecturales '
    'via CLIP ViT-B-32, PGVector & Agent LangGraph</div>',
    unsafe_allow_html=True,
)

tab_chat, tab_gallery, tab_retriever = st.tabs([
    "💬 Chat Agentique (ReAct)",
    "🖼️ Médias & Tableaux Extraits",
    "🔍 Recherche Vectorielle Directe",
])


# ------------------------------------------------------------------------------
# ONGLET 1 : Chat Agentique (ReAct avec LangGraph)
# ------------------------------------------------------------------------------
with tab_chat:
    st.markdown("### Posez votre question à l'Agent IA")

    col_q1, col_q2 = st.columns([3, 1])

    with col_q1:
        user_query_text = st.text_input(
            "Votre question textuelle :",
            placeholder="Ex : Quelle est la formule de l'attention Scaled Dot-Product ?",
            key="user_query_text",
        )

    with col_q2:
        query_image_file = st.file_uploader(
            "Ou uploadez une image :",
            type=["png", "jpg", "jpeg"],
            key="query_image_file",
            help="Fournissez un diagramme ou une figure pour une recherche par similarité visuelle CLIP.",
        )

    btn_submit = st.button("Envoyer à l'Agent IA", type="primary")

    if btn_submit:
        query_to_run = None
        img_temp_path = None

        if query_image_file is not None:
            temp_img_dir = Path("temp_queries")
            temp_img_dir.mkdir(exist_ok=True)
            img_temp_path = temp_img_dir / query_image_file.name
            with open(img_temp_path, "wb") as f:
                f.write(query_image_file.getbuffer())
            query_to_run = str(img_temp_path)
            st.image(Image.open(img_temp_path), caption="Image fournie en requête", width=250)
        elif user_query_text.strip():
            query_to_run = user_query_text.strip()

        if not query_to_run:
            st.warning("Veuillez saisir une question textuelle ou sélectionner une image.")
        else:
            with st.spinner("L'agent ReAct réfléchit et interroge l'espace latent multimodal..."):
                try:
                    from langchain_core.messages import AIMessage, ToolMessage, HumanMessage

                    agent = get_cached_agent(llm_choice)
                    result = agent.run(query_to_run)
                    trajectory = result.get("trajectory", [])
                    final_answer = result.get("final_answer", "")

                    # 1. Trajectoire ReAct (Thought -> Action -> Observation)
                    with st.expander("🧠 Voir la trajectoire de raisonnement ReAct (LangGraph)", expanded=True):
                        for step_i, msg in enumerate(trajectory, start=1):
                            if isinstance(msg, AIMessage) and msg.tool_calls:
                                st.markdown(f"**Étape {step_i} — 💭 RAISONNEMENT & DÉCISION (MIND) :**")
                                st.info(msg.content)
                                for tc in msg.tool_calls:
                                    st.code(f"Appel d'outil : {tc['name']}\nArguments : {tc['args']}", language="python")
                            elif isinstance(msg, ToolMessage):
                                st.markdown(f"**Étape {step_i} — 🔍 OBSERVATION DU CORPS (BODY / PGVector) :**")
                                st.text_area("Contenu retourné :", msg.content, height=180, disabled=True)

                    # 2. Réponse Finale de l'Agent
                    st.markdown("### ✨ Réponse Synthétisée de l'Agent IA")
                    st.markdown(final_answer)

                    # 3. Extraction et affichage visuel des images retrouvées dans le contexte
                    retriever = get_cached_retriever()
                    retrieved_items = retriever.retrieve(query_to_run, top_k=top_k_val)

                    image_items = [it for it in retrieved_items if it.get("type") == "image"]
                    if image_items:
                        st.markdown("#### 🖼️ Figures et Diagrammes associés retrouvés :")
                        cols = st.columns(min(len(image_items), 3))
                        for idx, img_item in enumerate(image_items):
                            col = cols[idx % 3]
                            p = img_item.get("image_path")
                            if p and Path(p).exists():
                                with col:
                                    st.image(
                                        Image.open(p),
                                        caption=f"Page {img_item.get('page')} — {img_item.get('caption')}\n(Score: {img_item.get('score'):.4f})",
                                        width="stretch",
                                    )

                except Exception as e:
                    st.error(f"Une erreur est survenue lors de l'exécution de l'agent : {e}")


# ------------------------------------------------------------------------------
# ONGLET 2 : Galerie des Médias & Tableaux Extraits
# ------------------------------------------------------------------------------
with tab_gallery:
    st.markdown("### Médias, Schémas et Tableaux extraits du document")
    
    img_dir = Path(OUTPUT_IMAGE_DIR)
    if img_dir.exists():
        img_files = list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpg"))
        if img_files:
            st.markdown(f"**Total images/figures détectées sur disque : {len(img_files)}**")
            cols = st.columns(2)
            for idx, img_path in enumerate(img_files):
                col = cols[idx % 2]
                with col:
                    st.image(Image.open(img_path), caption=img_path.name, width="stretch")
        else:
            st.info("Aucune image trouvée dans le dossier extracted_images/. Lancez l'indexation.")
    else:
        st.info("Le dossier des images n'existe pas encore. Lancez l'indexation.")


# ------------------------------------------------------------------------------
# ONGLET 3 : Recherche Vectorielle Directe (Sans LLM)
# ------------------------------------------------------------------------------
with tab_retriever:
    st.markdown("### Test direct de similarité cosinus (CLIP + PGVector)")
    direct_query = st.text_input("Recherche brute :", value="Figure 1: The Transformer model architecture")
    
    if st.button("Lancer la recherche vectorielle"):
        with st.spinner("Calcul des similarités cosinus..."):
            retriever = get_cached_retriever()
            results = retriever.retrieve(direct_query, top_k=top_k_val)

            for i, r in enumerate(results, start=1):
                st.markdown(f"**[Résultat #{i}] Type: `{r['type'].upper()}` | Page: `{r['page']}` | Similarité Cosinus: `{r['score']:.4f}`**")
                if r["type"] == "image":
                    p = r.get("image_path")
                    if p and Path(p).exists():
                        st.image(Image.open(p), width=300, caption=r.get("caption"))
                st.code(r["text"][:300] + ("..." if len(r["text"]) > 300 else ""), language="markdown")
                st.divider()

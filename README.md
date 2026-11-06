# 🤖 RAG Multimodal & Agentique (Textes, Tableaux & Images)

Application RAG (Retrieval-Augmented Generation) multimodale et agentique autonome conçue selon les spécifications techniques de **`Activite_RAG_Multimodal_Agentique.pdf`**. Elle est capable d'extraire, d'indexer et d'interroger conjointement du texte, des tableaux structurés et des images d'un document technique (*Attention Is All You Need* ou tout document technique de maintenance).

Le système repose sur un espace vectoriel latent commun **CLIP (ViT-B-32)**, un stockage vectoriel haute performance **PostgreSQL + PGVector**, une orchestration agentique **LangGraph** (Pattern ReAct), une génération locale via **Qwen 2.5 (Ollama)** et une interface interactive **Streamlit**.

---

## 🏗️ Architecture Globale du Projet

```
Rag_Mullti_Modale_agentique/
│
├── attention is all you need.pdf      # Document source de référence (11 pages)
├── docker-compose.yml                 # Services Docker : PostgreSQL 16 PGVector (5432) & Ollama (11434)
├── .env                               # Paramètres d'environnement (DB, CLIP, Ollama, API keys)
├── requirements.txt                   # Dépendances Python (CLIP, PyMuPDF, LangGraph, Ollama, Streamlit)
├── start_app.bat                      # Lanceur 1-clic de l'application Streamlit pour Windows
│
├── extracted_images/                  # Figures et schémas vectoriels extraits en haute résolution
│   ├── page_3_figure_1__the_transformer.png  # Diagramme d'architecture du Transformer
│   ├── page_4_figure_2___left__scaled_d.png  # Schéma Scaled Dot-Product & Multi-Head
│   └── ...
│
├── src/
│   ├── __init__.py
│   ├── config.py                      # Configurations centralisées (Dimensions 512, hôtes IPv4, tables)
│   ├── extractor.py                   # Extraction multimodale (Chunks LlamaIndex, Tableaux Markdown, Images PyMuPDF)
│   ├── storage.py                     # Initialisation PGVector et schéma relationnel (HNSW / Cosine)
│   ├── embeddings.py                  # Encodeur multimodal CLIP (ViT-B-32, 512 dimensions, normalisation L2)
│   ├── indexer.py                     # Indexation multimodale et gestion des empreintes SHA-256
│   ├── query_engine.py                # Recherche vectorielle cosinus cross-modale (requête Texte ou Image)
│   ├── tools.py                       # Outil agentique LangChain (@tool multimodal_retriever_tool)
│   └── agent.py                       # Cerveau ReAct LangGraph (MIND: Qwen 2.5 via Ollama <-> BODY: PGVector)
│
├── app.py                             # Interface graphique Streamlit interactive
├── chat_cli.py                        # Interface terminal interactive en ligne de commande
├── run_indexing.py                    # Script d'indexation initiale du document PDF
├── test_query.py                      # Tests de similarité cosinus (Requêtes texte et recherche par image)
├── test_agent.py                      # Validation complète des 4 scénarios ReAct
├── test_ollama_qwen_integration.py    # Test d'intégration Qwen 2.5 via Ollama local
└── NOTE_TECHNIQUE.md                  # Rapport technique détaillé et benchmarks
```

---

## 🎯 Récapitulatif des 6 Étapes du Cahier des Charges

| Étape | Intitulé | Technologies | Description & Rôle |
|---|---|---|---|
| **1** | **Sélection du document** | `attention is all you need.pdf` | Document technique de référence contenant du texte scientifique dense, 5 figures d'architecture complexes et des tableaux de résultats. |
| **2** | **Extraction et indexation** | `PyMuPDF`, `pdfplumber`, `tabulate`, `LlamaIndex` | Découpage du texte en fragments (`SentenceSplitter`), conversion des tableaux en Markdown et extraction des diagrammes. Calcul du hash SHA-256 pour éviter les doublons. |
| **3** | **Traitement des images & Espace Latent** | `clip-ViT-B-32` (`sentence-transformers`) | Projection conjointe du texte et des images dans un espace vectoriel partagé à 512 dimensions avec normalisation $L_2$. |
| **4** | **Question-réponse & Génération** | **Qwen 2.5** via **Ollama** (`ChatOllama`) | Modèle open source local exécuté sur `http://127.0.0.1:11434`. Synthèse ancrée (*grounded*) avec fallback dynamique sur Gemini si configuré. |
| **5** | **Interface utilisateur** | **Streamlit** (`app.py`) | Application web 3 onglets : Chat ReAct en direct, Galerie des médias extraits, Moteur de recherche vectoriel direct, et upload de nouveaux PDF. |
| **6** | **Orchestration agentique** | **LangGraph** (Pattern ReAct) | Boucle de raisonnement autonome (*Thought $\rightarrow$ Action $\rightarrow$ Observation $\rightarrow$ Final Answer*) articulant le Cerveau (LLM) et le Corps (`multimodal_retriever_tool`). |

---

## 🚀 Démarrage Rapide

### 1. Démarrer l'infrastructure Docker (PGVector + Ollama)
```bash
docker compose up -d
```
Les conteneurs suivants démarrent :
- `rag_pgvector` : `127.0.0.1:5432` (PostgreSQL 16 avec extension pgvector).
- `rag_ollama` : `127.0.0.1:11434` (Ollama avec le modèle `qwen2.5:1.5b` préchargé).

### 2. Indexer le document source dans PGVector
```bash
.\.venv\Scripts\python.exe run_indexing.py
```
> Le script extrait 29 nœuds multimodaux (23 fragments de texte, 1 tableau Markdown, 5 images/diagrammes) et les indexe dans la table `pdf_multimodal_clip_nodes`.

### 3. Lancer l'Application Streamlit
Double-cliquez sur `start_app.bat` ou lancez :
```bash
.\.venv\Scripts\python.exe -m streamlit run app.py
```
L'interface s'ouvre automatiquement dans votre navigateur sur **[http://localhost:8501](http://localhost:8501)**.

---

## 🧪 Validation & Tests en Ligne de Commande

### A. Tester la recherche vectorielle multimodale (CLIP + PGVector)
```bash
.\.venv\Scripts\python.exe test_query.py
```
Teste la recherche cosinus sur une requête textuelle et sur une recherche par image directe (`page_3_figure_1__the_transformer.png`, score de similarité 1.0000).

### B. Tester l'intégration locale Qwen 2.5 + Ollama
```bash
.\.venv\Scripts\python.exe test_ollama_qwen_integration.py
```
Valide la communication avec le conteneur Ollama et l'exécution complète d'un tour de raisonnement ReAct.

### C. Tester les scénarios complets de l'Agent IA
```bash
.\.venv\Scripts\python.exe test_agent.py --provider local --scenario 2
```
Scénarios disponibles :
1. Question mathématique / théorique (Formule d'attention).
2. Question architecturale (Figure 1 du Transformer).
3. Question tabulaire (Scores BLEU et hyperparamètres).
4. Question multimodale avec chemin d'image en entrée.

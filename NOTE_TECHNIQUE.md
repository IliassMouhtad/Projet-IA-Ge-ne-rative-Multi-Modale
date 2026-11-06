# Note Technique : RAG Multimodal & Agentique

**Projet :** Application RAG (Retrieval-Augmented Generation) multimodale et agentique sur documents techniques complexes (PDF).  
**Document documentaire de référence :** *Attention Is All You Need* (Vaswani et al.).  
**Auteur :** Antigravity AI Pair Programmer.  

---

## 1. Description des Modèles Choisis

### 1.1 Modèle d'Embeddings Multimodal : CLIP (ViT-B-32)
* **Architecture :** Vision Transformer (ViT-B/32) combiné à un encodeur textuel Transformer (OpenAI / OpenCLIP).
* **Espace vectoriel :** 512 dimensions.
* **Particularité essentielle :** Contrairement aux modèles d'embeddings textuels classiques (ex: BERT ou BGE), CLIP projette conjointement les **images (pixels réels)** et le **texte** dans le **même espace latent partagé**.
* **Normalisation :** Tous les vecteurs sont normalisés selon la norme $L_2$ ($\|\mathbf{v}\|_2 = 1$). Cela permet de calculer la similarité sémantique directement par le produit scalaire (similarité cosinus) :
  $$\text{Sim}(\mathbf{u}, \mathbf{v}) = \mathbf{u} \cdot \mathbf{v}$$

### 1.2 Modèle de Langage (LLM / Cerveau Agentique - MIND)
* **Modèle déployé localement :** **Qwen 2.5 (1.5B / 7B)** exécuté localement et de manière open source via **Ollama** (`rag_ollama` sur `http://127.0.0.1:11434`).
* **Optimisations CPU intégrées :**
  - Parallélisation multi-cœurs : `num_thread=4` exploitant tous les cœurs logiques de l'hôte.
  - Fenêtre de contexte maîtrisée : `num_ctx=1024` et `num_predict=220` pour empêcher les boucles de génération et accélérer l'inférence.
  - Compactage des extraits multimodaux : compression des snippets textuels à ~280 caractères, divisant la taille du prompt par 5.
  - Filtrage conversationnel immédiat : traitement sans latence (0.01s) des messages de politesse sans appel DB inutile.
* **Fournisseurs compatibles intégrés :**
  1. **Ollama local :** `qwen2.5:1.5b` ou `qwen2.5:7b` via `ChatOllama` (port `11434`), avec support natif du function calling et contrôle de génération (`num_predict=350`).
  2. **Groq API :** Modèles open source ultra-rapides (`llama-3.3-70b-versatile`, `qwen`).
  3. **OpenAI API :** `gpt-4o-mini`.
  4. **Moteur Local ReAct Autonome :** Moteur déterministe intégré pour garantir l'exécution instantanée du graphe et des tests de démonstration sans latence CPU.

---

## 2. Outils et Architecture Logicielle

| Composant | Rôle | Technologie retenue |
| :--- | :--- | :--- |
| **Extraction PDF** | Découpage textuel, détection de figures & conversion tabulaire Markdown | `LlamaIndex` (`SentenceSplitter`), `PyMuPDF` (`fitz`), `pdfplumber`, `tabulate` |
| **Espace Latent** | Vectorisation conjointe texte et image normalisée (512d) | `sentence-transformers` (`clip-ViT-B-32`), `open_clip_torch`, `torch`, `torchvision` |
| **Base Vectorielle** | Stockage persistant des vecteurs et indexation cosinus | `PostgreSQL 16` + extension `PGVector` (Docker `rag_pgvector`) |
| **Gestion des versions** | Détection des modifications et réindexation incrémentale | Empreinte cryptographique **SHA-256** du document PDF |
| **Outil Agentique** | Encapsulation de la recherche vectorielle multimodale | Décorateur `@tool` de `langchain_core` (`multimodal_retriever_tool`) |
| **Orchestrateur Agentique** | Cycle ReAct (*Thought* $\rightarrow$ *Action* $\rightarrow$ *Observation* $\rightarrow$ *Answer*) | **LangGraph** (`StateGraph`, `ToolNode`, `tools_condition`) |
| **Interface Utilisateur** | Visualisation, upload PDF, questions/réponses et affichage des diagrammes | **Streamlit** (`app.py`) |

---

## 3. Schéma Fonctionnel du Flux de Données

```
[DOCUMENT PDF]
      │
      ▼
[LlamaIndex + PyMuPDF + pdfplumber]
      │
      ├─► Chunks Textes ──────► Encodeur Texte CLIP ──┐
      ├─► Tableaux Markdown ──► Encodeur Texte CLIP ──┼─► [PGVector PostgreSQL (512d)]
      └─► Images & Figures ───► Encodeur Image CLIP ──┘          ▲
                                                                 │
[UTILISATEUR (Streamlit / CLI)]                                  │
      │                                                          │
      ▼                                                          │
[AGENT LANGGRAPH (ReAct)]                                        │
      │ 1. Thought (Raisonnement)                                │
      ▼                                                          │
[multimodal_retriever_tool (BODY)] ──────────────────────────────┘
      │ 2. Action & Observation (Récupération des contextes et figures)
      ▼
[LLM OPEN SOURCE (MIND: Qwen / Ollama)]
      │ 3. Synthèse contextualisée
      ▼
[RÉPONSE FINALE AVEC FIGURES & TABLEAUX AFFICHÉS]
```

---

## 4. Limites Actuelles et Pistes d'Amélioration

1. **Résolution native de CLIP (224x224 pixels) :**  
   Pour des schémas techniques très denses ou des schémas électriques complexes, un redimensionnement à 224x224 peut flouter les petits textes à l'intérieur des figures.  
   *Piste d'amélioration :* Utiliser des modèles récents à plus haute résolution comme **CLIP ViT-L-14@336px** ou **SigLIP**.

2. **Descriptions textuelles des images générées par VLM :**  
   Actuellement, les descriptions des images s'appuient sur les légendes extraites par PyMuPDF et les pixels bruts encodés par CLIP.  
   *Piste d'amélioration :* Brancher un modèle Vision-Language Model léger (VLM) tel que **Moondream2**, **LLaVA-v1.6** ou **Qwen2-VL** pour générer automatiquement une explication textuelle détaillée de chaque schéma dès l'indexation.

3. **Dépendance matérielle pour les LLM locaux :**  
   L'exécution d'un modèle 7B/14B sur CPU peut introduire une latence de quelques secondes par réponse. Le système intègre un mode de repli autonome instantané pour pallier cette contrainte en démonstration.

---

## 5. Exemples de Questions / Réponses et Trajectoires ReAct

### Exemple 1 : Recherche Théorique et Mathématique
* **Requête utilisateur :** *"Quelle est la formule de l'attention Scaled Dot-Product et quel est le rôle du facteur de mise à l'échelle ?"*
* **Raisonnement ReAct :**  
  `[Thought]` La question demande la définition mathématique de l'attention. Appel de `multimodal_retriever_tool` avec les termes `"Scaled Dot-Product Attention formula"`.
* **Observation :**  
  Récupération du chunk page 4 :
  $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
  Explication du facteur $\frac{1}{\sqrt{d_k}}$ : pour les grandes dimensions $d_k$, le produit scalaire grandit en magnitude, poussant la fonction softmax vers des régions à gradients extrêmement faibles.
* **Réponse finale :** Réponse synthétisée et sourcée page 4.

### Exemple 2 : Recherche Visuelle & Diagramme
* **Requête utilisateur :** *"Montre-moi l'architecture du modèle Transformer."*
* **Raisonnement ReAct :**  
  `[Thought]` Détection d'un besoin de diagramme architectural. Appel de `multimodal_retriever_tool("Transformer model architecture diagram", top_k=3)`.
* **Observation :**  
  Récupération de la référence :  
  `Type: IMAGE | Page: 3 | Fichier: extracted_images/page_3_figure_1__the_transformer.png | Score: 0.85`
* **Réponse finale :** Affichage de l'explication des 6 couches de l'Encodeur et du Décodeur, accompagnée de l'affichage direct de l'image de la Figure 1.

### Exemple 3 : Recherche par Image Directe
* **Requête utilisateur :** `extracted_images/page_4_figure_2___left__scaled_d.png` (upload d'image).
* **Raisonnement ReAct :**  
  `[Thought]` Détection d'un fichier image. Encodage direct par le sous-système de vision CLIP et recherche dans PGVector.
* **Observation :**  
  `[Résultat #1] Type: IMAGE | Page: 4 | Figure 2: Scaled Dot-Product Attention | Score: 1.0000`  
  `[Résultat #2] Type: IMAGE | Page: 3 | Figure 1: The Transformer | Score: 0.8984`
* **Réponse finale :** Identification immédiate du sous-composant d'attention et association avec son implémentation dans le document.


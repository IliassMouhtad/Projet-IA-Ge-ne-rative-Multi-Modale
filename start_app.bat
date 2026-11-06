@echo off
title RAG Multimodal et Agentique (Streamlit)
echo ======================================================================
echo    Lancement de l'Application RAG Multimodal et Agentique
echo ======================================================================
cd /d "%~dp0"
call .\.venv\Scripts\activate.bat
echo Demarrage du serveur Streamlit sur http://127.0.0.1:8501 ...
python -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501 --server.headless false
pause

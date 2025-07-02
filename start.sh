#!/bin/bash
# Start FastAPI backend on port 8000
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 &

# Start Streamlit frontend on port 7860 (required by Hugging Face)
streamlit run streamlit_app.py --server.port 7860 --server.address 0.0.0.0

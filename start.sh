#!/bin/bash

# Start FastAPI backend in the background
uvicorn api.main:app --host 0.0.0.0 --port 8000 &

# Start Streamlit frontend
streamlit run streamlit_app.py --server.port=10000 --server.address=0.0.0.0

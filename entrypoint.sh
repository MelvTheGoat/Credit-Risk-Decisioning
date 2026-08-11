#!/bin/bash

# Start FastAPI on internal port 8000 in the background
uvicorn src.api.main:app --host 127.0.0.1 --port 8000 &

# Start Streamlit on public port 8080 in the foreground
streamlit run app.py --server.port 8080 --server.address 0.0.0.0

#!/bin/bash

# Start FastAPI on port 8000 (0.0.0.0 allows the GitHub CI tests to reach it)
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 &

# Start Streamlit on public port 8080
streamlit run app.py --server.port 8080 --server.address 0.0.0.0

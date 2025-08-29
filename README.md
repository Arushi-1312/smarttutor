# SmartTutor — Final Year Project Demo

SmartTutor demonstrates an AI pipeline combining data analysis, ML model training, optimization scheduling, and chatbot/LLM integration.

## Features
- Synthetic student dataset + EDA (pandas, NumPy)
- Predictive model for "dropout risk / performance" using scikit-learn (Logistic Regression / RandomForest)
- Optimization of tutorial scheduling using PuLP and OR-Tools
- Lightweight FastAPI server exposing endpoints for predictions and optimized schedules
- Rasa chatbot skeleton to answer queries and use the model via an action
- LLM inference example using Hugging Face (for summarization / answer generation)

## Requirements
- Python 3.10+
- See `requirements.txt`

## Quickstart
1. Create venv:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt

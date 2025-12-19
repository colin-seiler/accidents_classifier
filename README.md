# Accident Severity Prediction — End-to-End ML System

A **production-style, end-to-end machine learning application** that predicts accident severity using a trained ML pipeline, exposed through a FastAPI backend and an interactive Streamlit frontend — fully containerized with Docker.

This project demonstrates **full-stack ML engineering**, from model inference to service orchestration and deployment.

---

## Highlights

- Built and deployed a **complete ML system**, not just a notebook
- Served real-time predictions via a **REST API (FastAPI)**
- Designed a **user-facing Streamlit app** for interactive inference
- Fully **Dockerized & orchestrated with Docker Compose**
- Clean service separation and internal networking
- Ready for **cloud deployment (AWS / DigitalOcean / GCP)**

---

## Architecture

Streamlit UI  –>  FastAPI  –>  Trained ML Model

- Stateless API for scalable inference
- Services communicate over an isolated Docker bridge network
- Only the frontend is publicly exposed

---

## Machine Learning

- Model trained offline and serialized with `joblib`
- Uses a **scikit-learn pipeline** with feature preprocessing
- Gradient boosting classifier (LightGBM)
- Runtime model loading with health checks
- Robust input validation using Pydantic

---

## Tech Stack

**ML / Data**
- Python, pandas, scikit-learn, LightGBM

**Backend**
- FastAPI
- Pydantic
- Uvicorn

**Frontend**
- Streamlit

**DevOps / Deployment**
- Docker
- Docker Compose
- Linux (containerized runtime)

---

## Requirements

- Docker
- Docker Compose

No local Python environment is required.

---

## Running the Application

From the project root:

```bash
docker compose up --build
```
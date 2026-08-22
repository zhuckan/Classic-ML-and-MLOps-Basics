# Classic ML + MLOps Basics

This project demonstrates how to transform a "dirty" Jupyter notebook into a reproducible ML pipeline and REST API service.  
It uses the **Credit Card Fraud Detection** dataset and covers the full MLOps cycle: data versioning, experiment tracking, model training, and deployment.

## Key Features

- **DVC** — data versioning (datasets are not stored in Git)
- **MLflow** — tracking hyperparameters, metrics, and model artifacts
- **Classic ML models** — Random Forest / XGBoost
- **FastAPI** — REST API for serving the best model
- **Docker** / **docker-compose** — containerized deployment

## Tech Stack

`Python` · `DVC` · `MLflow` · `Scikit-learn` / `XGBoost` · `FastAPI` · `Docker`

## Getting Started

### Prerequisites
- Python 3.10+
- Git
- Docker (optional)

### Setup

1. **Clone the repository**
  ``` bash
  git clone https://github.com/zhuckan/Classic-ML-and-MLOps-Basics.git
  cd Classic-ML-and-MLOps-Basics
  ```
2. **Install dependencies**

  ``` bash
  pip install -r requirements.txt
  ```
3. **Download the dataset (via DVC)**

  ``` bash
  dvc pull
  ```
4. **Run the training pipeline**

  ``` bash
  python task1.py
  ```
5. **Start the REST API**

  ``` bash
  uvicorn app:app --reload
  ```
  The API will be available at http://127.0.0.1:8000.

**Run with Docker**
``` bash
docker-compose up --build
```

## Project Structure

| File | Description |
|------|-------------|
| `task1.py` | Main training pipeline |
| `app.py` | FastAPI application |
| `creditcard.csv.dvc` | DVC metadata file for the dataset |
| `Dockerfile`, `docker-compose.yml` | Containerization files |
| `requirements.txt` | Python dependencies |
| `task_1(Classic ML + MLOps Basics).ipynb` | Original notebook (for reference) |

# House Price Prediction — ML System (Assignment II)

This repository contains the refactored, production-ready Machine Learning system for predicting median house prices. Built as an extension of Assignment I, this version incorporates Object-Oriented Programming (OOP) design, modularized API routers, structured logging, defensive error handling, comprehensive `pytest` suites, and code quality linting.

---

## 📌 Project Architecture & Directory Structure

```text
house_price_project/
├── api/                    # Production REST API package (FastAPI)
│   ├── routers/            # Modular endpoint routing
│   │   ├── health.py       # Service health/liveness router
│   │   └── predict.py     # Inference execution router
│   ├── config.py           # API settings & path configurations
│   ├── dependencies.py     # Shared dependencies (Singleton Predictor)
│   ├── schemas.py          # Pydantic request/response validation schemas
│   └── main.py             # FastAPI entry point & application assembly
├── src/                    # Core ML source code package
│   ├── ingest.py           # DataIngestion class with error handling
│   ├── pipeline.py         # DataPreparationPipeline class
│   ├── model_training.py   # ModelTrainer class (Linear Regression & Random Forest)
│   ├── predictor.py        # PricePredictor inference engine
│   └── logger.py           # Centralized logging setup
├── tests/                  # Automated pytest test suite
│   ├── test_pipeline.py    # Unit & data validation tests
│   └── test_model.py       # ML-specific training & inference tests
├── data/                   # Data storage directory
│   └── housing.csv         # Raw input dataset
├── models/                 # Auto-created serialized model artifacts (.joblib, .json)
├── figures/                # Auto-created evaluation charts & plot outputs
├── requirements.txt        # Managed package dependencies
└── README.md               # Project documentation

```

---

## 🚀 Getting Started

### 1. Prerequisites

* Python `3.10` or higher


* `pip` or `conda` package manager

### 2. Installation & Environment Setup

Clone the repository and navigate to the project root directory:

```bash
git clone <repository-url>
cd house_price_project

```

Create and activate a virtual environment:

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

```

Install all required dependencies:

```bash
pip install -r requirements.txt

```

---

## 🛠️ Usage Guidelines

### 1. Execute the Data Preparation & Model Training Pipeline

Run the training pipeline to clean data, engineer features, train models (Linear Regression & Random Forest), and save evaluation metrics and artifacts to `models/` and `figures/`:

```bash
python -m src.model_training

```

### 2. Launch the Production REST API

Start the FastAPI microservice using Uvicorn:

```bash
uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload

```

* **Interactive API Documentation (Swagger UI):** Open `[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)` in your browser.
* **Health Check Endpoint:** `GET [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)`
* **Inference Endpoint:** `POST [http://127.0.0.1:8000/predict](http://127.0.0.1:8000/predict)`

#### Sample JSON Payload for `/predict`:

```json
{
  "longitude": -122.23,
  "latitude": 37.88,
  "housing_median_age": 41.0,
  "total_rooms": 880.0,
  "total_bedrooms": 129.0,
  "population": 322.0,
  "households": 126.0,
  "median_income": 8.3252,
  "ocean_proximity": "NEAR BAY"
}

```

---

## 🧪 Quality Assurance & Testing

### 1. Running Automated Tests with `pytest`

Execute the full suite of unit, integration, and ML-specific tests:

```bash
pytest tests/ -v

```

**Test Coverage Highlights:**

* **Data Validation:** Imputation checks for missing values and derived column validations.


* **Model Integrity:** Small batch overfitting checks (capacity verification).


* **Inference Boundaries:** Prediction value bounds checks and non-negativity guarantees.



### 2. Code Linting & Static Analysis

Check adherence to PEP 8 standards using `pylint`:

```bash
pylint src/ api/ tests/

```

---

## 🔒 Security & Deployment Considerations

* **Input Schema Validation:** All request bodies strictly adhere to Pydantic field constraints ($ge=0$, $gt=0$) to protect against adversarial inputs and invalid payload injection.


* **Shadow Deployment Strategy:** Designed to support zero-downtime production deployment by duplicating live traffic to compare candidate model versions against current baselines asynchronously.

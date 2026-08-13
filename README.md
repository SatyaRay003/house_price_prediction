Here is the complete, updated **`README.md`** file. It includes the **Project Architecture & Directory Structure** matching your attached screenshot, along with the **Execution Commands Summary** section.

---

# House Price Prediction — ML System (Assignment II)

This repository contains the refactored, production-ready Machine Learning system for predicting median house prices. Built as an extension of Assignment I, this version incorporates Object-Oriented Programming (OOP) design, modularized API routers, structured logging, defensive error handling, comprehensive `pytest` suites, and code quality linting.

---

## 📌 Project Architecture & Directory Structure

```text
house_price_prod/
├── api/                    # Production REST API package (FastAPI)
│   ├── routers/            # Modular endpoint routing
│   │   ├── __init__.py
│   │   ├── health.py       # Service health/liveness router
│   │   └── predict.py      # Inference execution router
│   ├── __init__.py
│   ├── config.py           # API settings & path configurations
│   ├── dependencies.py     # Shared dependencies (Singleton Predictor)
│   ├── main.py             # FastAPI entry point & application assembly
│   └── schemas.py          # Pydantic request/response validation schemas
├── app/                    # Primary application package
│   ├── __init__.py         # Package initialization
│   ├── data_ingestion.py   # Data Ingestion class & exception handling
│   ├── feature_engineering.py  # Data cleaning & feature transformation pipeline
│   ├── logger.py           # Centralized logging configuration
│   ├── model_inference.py  # Inference engine for serving predictions
│   └── model_training.py   # Model training, persistence & visualization
├── data/                   # Data storage directory
│   └── housing.csv         # Raw input dataset
├── figures/                # Auto-created evaluation charts & plot outputs
├── models/                 # Auto-created serialized model artifacts (.joblib, .json)
├── tests/                  # Automated pytest test suite
│   ├── test_pipeline.py    # Unit & data validation tests
│   └── test_model.py       # ML-specific training & inference tests
├── .gitignore              # Git ignore rules
├── README.md               # Project documentation
├── requirements.txt        # Managed package dependencies
├── run_api.py              # Root launcher script for REST API
├── training_pipeline.py    # Root orchestration script for offline ML pipeline
└── utils.py                # Helper utilities

```

---

## ⚙️ Execution Commands Summary

Below is a quick reference table of all terminal commands required to set up, train, serve, test, and lint the project from the root directory:

| Task / Purpose | Command | Description |
| :--- | :--- | :--- |
| **Install Dependencies** | `pip install -r requirements.txt` | Installs all required Python packages. |
| **Run ML Training Pipeline** | `python training_pipeline.py` | Executes ingestion, feature engineering, model training, persistence, and figure generation. |
| **Launch Production REST API** | `python run_api.py` | Starts the Uvicorn web server hosting FastAPI endpoints on `http://127.0.0.1:8000`. |
| **Run Test Suite** | `pytest tests/ -v` | Runs unit, integration, and ML component validation tests via Pytest. |
| **Lint Codebase** | `pylint app/ api/ tests/` | Performs PEP 8 code quality analysis and static lint checks. |

---

## 🚀 Getting Started

### 1. Prerequisites

* Python `3.10` or higher


* `pip` or `conda` package manager

### 2. Installation & Environment Setup

Clone the repository and navigate to the project root directory:

```bash
git clone <repository-url>
cd house_price_prod

```

Create and activate a virtual environment:

```bash
# Using venv
python -m venv house_price_env
source house_price_env/bin/activate  # On Windows: house_price_env\Scripts\activate

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
python training_pipeline.py

```

### 2. Launch the Production REST API

Start the FastAPI microservice using the root launcher script:

```bash
python run_api.py

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
pylint app/ api/ tests/

```

---

## 🔒 Security & Deployment Considerations

* **Input Schema Validation:** All request bodies strictly adhere to Pydantic field constraints ($ge=0$, $gt=0$) to protect against adversarial inputs and invalid payload injection.


* **Shadow Deployment Strategy:** Designed to support zero-downtime production deployment by duplicating live traffic to compare candidate model versions against current baselines asynchronously.



---

# MLOps Pipeline for Electricity Demand Forecasting ⚡️

This project implements a robust, end-to-end Machine Learning Operations (MLOps) pipeline designed to accurately forecast **next-hour electricity demand**. The primary goal is to address the critical business challenge of **resource misallocation** within the energy sector, which leads to increased operational costs, a higher carbon footprint, and reduced efficiency. By providing precise demand predictions, the system enables optimal resource planning, ultimately enhancing grid reliability and promoting sustainable energy management.

---

## Business Problem Addressed

The energy company faces significant challenges due to inefficient distribution of energy resources. This results in:
* **Increased Operational Costs:** Over- or under-generation of electricity leads to wasteful spending.
* **Higher Carbon Footprint:** Inefficient generation often means burning more fossil fuels than necessary.
* **Reduced Efficiency:** Mismatch between supply and demand impacts grid stability and responsiveness.

This project aims to directly mitigate these issues by providing a proactive forecasting solution.

---

## ML Problem Statement

To enable effective resource allocation, the core machine learning problem is to **build a model that accurately predicts next-hour electricity demand**. This prediction allows the company to align energy generation with forecasted needs, ensuring:
* **Optimal Resource Allocation:** Generating just enough energy to meet demand.
* **Cost Reduction:** Minimizing fuel consumption and operational overhead.
* **Improved Reliability:** Preventing power shortages or overloads.

---

## Data Sources

Accurate demand forecasting relies on integrating diverse data streams:
* **Historical Electricity Demand:** Retrieved from the **EIA API** (U.S. Energy Information Administration).
* **Weather Data:** Sourced from the **Open-Meteo Weather API** (temperature, humidity, etc.).
* **Calendar Events:** Public holidays extracted using `pandas.tseries.holiday` to capture demand fluctuations around special dates.

---

## Project Methodology: The MLOps Pipeline

This project is structured around a **three-stage MLOps pipeline**, embodying key principles such as modular architecture, centralized feature stores, model registries, and automated inference.

### 1. Feature Pipeline 📊

This stage focuses on **data ingestion, transformation, and feature engineering**, preparing the raw data into a format suitable for machine learning models.

**Key Steps:**
* **Data Fetching:** Programmatically retrieves electricity demand data from the **EIA API** and corresponding temperature data from the **Open-Meteo Weather API**.
* **Data Merging:** Integrates both datasets based on their timestamps, ensuring a unified view.
* **Feature Engineering:** Transforms the raw data into rich time-series features crucial for predictive accuracy:
    * **Lag Features:** Creates lagged versions of electricity demand (e.g., demand from 1 hour ago, 24 hours ago, 7 days ago).
    * **Rolling Statistics:** Computes moving averages and standard deviations over various time windows (e.g., 24-hour rolling mean).
    * **Temporal Features:** Extracts features like `hour_of_day`, `day_of_week`, `month`, and `year`.
    * **Holiday Indicators:** Generates binary flags for public holidays to capture unique demand patterns.
* **Feature Storage:** Stores the meticulously engineered features in the **Hopsworks Feature Store**, ensuring data consistency, versioning, and discoverability for subsequent stages.

### 2. Training Pipeline 🧠

This stage is dedicated to **model development, hyperparameter tuning, and model registration**, ensuring the creation of a high-performing and well-managed forecasting model.

**Key Steps:**
* **Feature Loading:** Retrieves the processed features and the defined target variable (next-hour electricity demand) directly from the Hopsworks Feature Store.
* **Model Training:** A **LightGBM** model is trained, selected for its efficiency and strong performance in tabular and time-series data.
* **Hyperparameter Tuning:** **Optuna** is employed for automated hyperparameter optimization, systematically exploring different configurations to minimize the **Mean Absolute Error (MAE)**.
* **Cross-Validation:** Implements a robust 5-fold time-series cross-validation strategy to evaluate model performance, ensuring generalizability and preventing overfitting.
* **Model Registration:** The best-performing model, along with its metadata and metrics, is versioned and saved to the **Hopsworks Model Registry**, making it ready for deployment and easy to track.

### 3. Inference Pipeline 🔮

This stage focuses on **generating hourly forecasts** using the trained model, ensuring continuous and up-to-date predictions.

**Key Steps:**
* **Automated Scheduling:** The inference process is orchestrated via **GitHub Actions**, running as a serverless Python script (`inference_pipeline.py`) on an hourly cron schedule (`cron: '0 * * * *'`).
* **Latest Feature Retrieval:** Fetches the most recent features required for prediction from the Hopsworks Feature Store, ensuring the forecasts are based on current data.
* **Model Loading:** Dynamically loads the latest registered model from the Hopsworks Model Registry.
* **Prediction Generation:** Utilizes the loaded model to predict the next-hour electricity demand.
* **Performance Monitoring:** Computes the MAE against actual demand (when available) and logs these metrics, contributing to the real-time monitoring dashboard.

---

## Deployment & Monitoring

To provide actionable insights and track system performance, two interactive applications were developed using **Streamlit**:

* **Batch Forecasting App:** An interactive web application that visualizes predicted hourly electricity demand across various NYC regions on a map-based interface.
* **Monitoring Dashboard:** A dynamic dashboard that provides real-time visibility into the model's performance. It features trend lines of the **Mean Absolute Error (MAE)**, historical performance insights, and visual comparisons between predicted and actual electricity demand, empowering data-driven decisions.

---

## Technologies Used

* **Programming Language:** Python
* **Data Manipulation & Analysis:** `pandas`, `numpy`
* **Machine Learning Frameworks:** `scikit-learn`, `lightgbm`
* **Hyperparameter Optimization:** `optuna`
* **MLOps Platform:** **Hopsworks** (for Feature Store and Model Registry)
* **API Integration:** `requests` (for EIA API, Open-Meteo Weather API)
* **Orchestration & CI/CD:** **GitHub Actions**
* **Web Application & Visualization:** **Streamlit**

---

## Project Highlights

* **Comprehensive MLOps Pipeline:** Developed a full-cycle, automated ML pipeline covering data ingestion, feature engineering, model training, deployment, and monitoring.
* **Production-Ready System:** Architected for scalability, reproducibility, and reliability, capable of continuous hourly inference.
* **Business Impact:** Directly addresses critical operational challenges by enabling optimized resource allocation, cost reduction, and environmental sustainability.
* **Interactive Insights:** Delivered a user-friendly Streamlit dashboard for real-time model performance tracking and actionable demand forecasts.

---

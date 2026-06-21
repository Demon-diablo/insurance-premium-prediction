# Insurance Premium Prediction

<p align="center">
  <img src="images/app_preview.png" width="900">
</p>

An end-to-end Machine Learning project that predicts medical insurance premiums using Linear Regression. The project includes data preprocessing, exploratory data analysis (EDA), model training, evaluation, and an interactive Streamlit web application for real-time predictions.

## Highlights

- End-to-end ML workflow from raw data to deployed web app
- Exploratory Data Analysis (EDA) with visual insights
- Feature preprocessing pipeline using `scikit-learn` `Pipeline` + `ColumnTransformer`
- Linear Regression model trained on real-world insurance data
- Interactive Streamlit application for real-time premium predictions
- Model evaluation with R², RMSE, and MAE metrics

## Workflow

1. Load dataset
2. Perform EDA
3. Clean data (remove duplicates)
4. Train-test split
5. Build preprocessing pipeline (`StandardScaler` + `OneHotEncoder`)
6. Train Linear Regression model
7. Evaluate model performance
8. Save pipeline with joblib
9. Deploy with Streamlit

## Dataset

- **Source:** [Medical Insurance Cost Dataset](https://www.kaggle.com/datasets/mosapabdelghany/medical-insurance-cost-dataset) (Kaggle)
- **Original records:** 1,338
- **Final records after cleaning:** 1,337 (1 duplicate removed)
- **Features:** age, sex, bmi, children, smoker, region
- **Target:** insurance charges

## Model Performance

| Metric | Score |
|--------|-------|
| R²     | 0.796 |
| RMSE   | $5,940 |
| MAE    | $4,069 |

## Tech Stack

| Category       | Technologies                          |
|----------------|---------------------------------------|
| Language       | Python                                |
| ML             | scikit-learn                          |
| Data           | pandas, NumPy                         |
| Visualization  | Matplotlib, Seaborn                   |
| Web App        | Streamlit                             |
| Serialization  | Joblib                                |

## Project Structure

```text
insurance-premium-prediction-Linear/
├── app/
│   └── app.py                  # Streamlit web interface
├── data/
│   ├── raw/insurance.csv       # Dataset
│   └── processed/              # (processed data output)
├── images/                     # (figures and previews)
├── models/
│   └── linear_regression_pipeline.pkl  # Trained model pipeline
├── notebooks/
│   └── insurance_eda_model.ipynb       # EDA + model training
├── requirements.txt
├── .gitignore
└── README.md
```

## Quick Start

```bash
# Clone the repository
git clone <repo-url>
cd insurance-premium-prediction-Linear

# Create virtual environment and install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app/app.py

# Or re-run the notebook to retrain the model
jupyter notebook notebooks/insurance_eda_model.ipynb
```

## Future Improvements

- Ridge, Lasso, and ElasticNet regression for comparison
- Model comparison dashboard
- SHAP explainability for feature importance
- Hyperparameter tuning with GridSearchCV
- Deployment on cloud (Hugging Face Spaces / Streamlit Cloud)

## Acknowledgements

- Dataset: [Medical Insurance Cost Dataset](https://www.kaggle.com/datasets/mosapabdelghany/medical-insurance-cost-dataset) by Mosap Abdelghany (Kaggle)

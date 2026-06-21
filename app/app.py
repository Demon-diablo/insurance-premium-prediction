import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import matplotlib.pyplot as plt

# ------------------ Page Config ------------------
st.set_page_config(
    page_title="Insurance Premium Prediction",
    page_icon="💰",
    layout="wide"
)

# ------------------ Load Model & Data (Cached) ------------------
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "linear_regression_pipeline.pkl"
DATA_PATH = BASE_DIR / "data" / "raw" / "insurance.csv"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

try:
    model = load_model()
    dataset = load_data()
except FileNotFoundError as e:
    st.error(f"Required file not found: {e}")
    st.stop()
except Exception as e:
    st.error(f"Error loading model or dataset: {e}")
    st.stop()

# ------------------ Sidebar ------------------
st.sidebar.title("⚙️ Prediction Settings")
st.sidebar.header("Input Features")

age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=25)
sex = st.sidebar.selectbox("Gender", ["male", "female"])

bmi = st.sidebar.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
if bmi < 15 or bmi > 50:
    st.sidebar.caption("⚠️ This BMI is outside typical ranges.")

children = st.sidebar.number_input("Children", min_value=0, max_value=10, value=0)
smoker = st.sidebar.selectbox("Smoker", ["yes", "no"])
region = st.sidebar.selectbox("Region", ["southwest", "southeast", "northwest", "northeast"])

predict = st.sidebar.button("Predict Premium")

# ------------------ Main Page ------------------
st.title("💰 Insurance Premium Prediction")
st.write("Predict medical insurance charges using a trained Linear Regression model.")

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Model Information")
    st.info("""
**Model:** Linear Regression

**Dataset:** Medical Insurance Cost Dataset

**Target Variable:** Insurance Charges
""")

with col2:
    st.subheader("Prediction")

    if predict:
        input_df = pd.DataFrame({
            "age": [age],
            "sex": [sex],
            "bmi": [bmi],
            "children": [children],
            "smoker": [smoker],
            "region": [region]
        })

        try:
            prediction = model.predict(input_df)[0]
            st.metric(
                label="Estimated Insurance Charges",
                value=f"${prediction:,.2f}"
            )
        except Exception as e:
            st.error(f"Prediction failed: {e}")
    else:
        st.warning("Enter the details in the sidebar and click **Predict Premium**.")

st.divider()

st.subheader("Input Summary")
st.dataframe(
    pd.DataFrame({
        "Feature": ["Age", "Gender", "BMI", "Children", "Smoker", "Region"],
        "Value": [age, sex, bmi, children, smoker, region]
    }),
    use_container_width=True
)

st.divider()

# ------------------ Tabs ------------------
tab1, tab2, tab3 = st.tabs(["📄 Dataset", "📊 Visualizations", "ℹ️ About Model"])

# ------------------ Dataset ------------------
with tab1:
    option = st.selectbox(
        "Select View",
        ["First 10 Rows", "Dataset Information", "Statistical Summary", "Full Dataset"]
    )

    if option == "First 10 Rows":
        st.dataframe(dataset.head(10), use_container_width=True)

    elif option == "Dataset Information":
        info = pd.DataFrame({
            "Column": dataset.columns,
            "Data Type": dataset.dtypes.astype(str),
            "Missing Values": dataset.isnull().sum().values
        })
        st.dataframe(info, use_container_width=True)

    elif option == "Statistical Summary":
        st.dataframe(dataset.describe(), use_container_width=True)

    else:
        st.dataframe(dataset, use_container_width=True)

# ------------------ Visualizations ------------------
with tab2:
    plot = st.selectbox(
        "Choose Visualization",
        ["Age vs Charges", "BMI vs Charges", "Smoker vs Charges", "Region Distribution", "Charges Distribution"]
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    if plot == "Age vs Charges":
        ax.scatter(dataset["age"], dataset["charges"], alpha=0.6)
        ax.set_xlabel("Age")
        ax.set_ylabel("Charges")
        ax.set_title("Age vs Charges")

    elif plot == "BMI vs Charges":
        ax.scatter(dataset["bmi"], dataset["charges"], alpha=0.6)
        ax.set_xlabel("BMI")
        ax.set_ylabel("Charges")
        ax.set_title("BMI vs Charges")

    elif plot == "Smoker vs Charges":
        dataset.boxplot(column="charges", by="smoker", ax=ax)
        ax.set_title("Smoker vs Charges")
        ax.set_xlabel("Smoker")
        ax.set_ylabel("Charges")
        fig.suptitle("")  # remove default pandas title

    elif plot == "Region Distribution":
        dataset["region"].value_counts().plot(kind="bar", ax=ax, color="skyblue")
        ax.set_xlabel("Region")
        ax.set_ylabel("Count")
        ax.set_title("Region Distribution")

    elif plot == "Charges Distribution":
        ax.hist(dataset["charges"], bins=25, color="salmon", edgecolor="black")
        ax.set_xlabel("Charges")
        ax.set_ylabel("Frequency")
        ax.set_title("Distribution of Charges")

    plt.tight_layout()
    st.pyplot(fig)

# ------------------ About Model ------------------
with tab3:
    st.subheader("Model Details")

    st.markdown("""
**Algorithm:** Linear Regression

**Target Variable:** Charges

**Features Used**
- Age
- Gender
- BMI
- Children
- Smoker
- Region
""")

    metric1, metric2, metric3 = st.columns(3)
    metric1.metric("R² Score", "0.767")
    metric2.metric("RMSE", "5926")
    metric3.metric("MAE", "4244")
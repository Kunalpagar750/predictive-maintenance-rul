
import streamlit as st
import pandas as pd
import joblib


# -----------------------------
# Page configuration
# -----------------------------

st.set_page_config(
    page_title="RUL Prediction",
    page_icon="✈️",
    layout="wide"
)


# -----------------------------
# Load trained model
# -----------------------------

model_package = joblib.load("rul_model.pkl")

model = model_package["model"]
feature_columns = model_package["features"]


# -----------------------------
# Load test data
# -----------------------------

columns = [
    "unit_id",
    "cycle",
    "setting_1",
    "setting_2",
    "setting_3",
    "sensor_1",
    "sensor_2",
    "sensor_3",
    "sensor_4",
    "sensor_5",
    "sensor_6",
    "sensor_7",
    "sensor_8",
    "sensor_9",
    "sensor_10",
    "sensor_11",
    "sensor_12",
    "sensor_13",
    "sensor_14",
    "sensor_15",
    "sensor_16",
    "sensor_17",
    "sensor_18",
    "sensor_19",
    "sensor_20",
    "sensor_21"
]

test_df = pd.read_csv(
    "data/test_FD001.txt",
    sep=r"\s+",
    header=None
)

test_df.columns = columns


# -----------------------------
# Get final observation
# for each engine
# -----------------------------

test_last = (
    test_df
    .sort_values(["unit_id", "cycle"])
    .groupby("unit_id")
    .last()
    .reset_index()
)
# -----------------------------
# Load actual RUL values
# -----------------------------

rul_path = "data/RUL_FD001.txt"

actual_rul_df = pd.read_csv(
    rul_path,
    sep=r"\s+",
    header=None,
    names=["actual_RUL"]
)

actual_rul_df["unit_id"] = range(
    1,
    len(actual_rul_df) + 1
)


# -----------------------------
# Remove constant columns
# -----------------------------

constant_columns = [
    "setting_3",
    "sensor_1",
    "sensor_5",
    "sensor_10",
    "sensor_16",
    "sensor_18",
    "sensor_19"
]

test_last = test_last.drop(columns=constant_columns)


# -----------------------------
# Dashboard title
# -----------------------------

st.title("✈️ Predictive Maintenance — RUL Estimation")

st.write(
    "Predict the Remaining Useful Life (RUL) of a simulated aircraft engine "
    "using sensor and operating-condition data."
)


st.divider()


# -----------------------------
# Engine selection
# -----------------------------

engine_ids = test_last["unit_id"].tolist()

selected_engine = st.selectbox(
    "Select an Engine",
    engine_ids
)


# -----------------------------
# Get selected engine data
# -----------------------------

engine_data = test_last[
    test_last["unit_id"] == selected_engine
].copy()


# Keep exactly the features
# expected by the model

X_engine = engine_data[feature_columns]


# -----------------------------
# Prediction
# -----------------------------

prediction = model.predict(X_engine)[0]

prediction = max(0, prediction)

# -----------------------------
# Actual RUL and prediction error
# -----------------------------

actual_rul = actual_rul_df.loc[
    actual_rul_df["unit_id"] == selected_engine,
    "actual_RUL"
].iloc[0]

prediction_error = prediction - actual_rul
absolute_error = abs(prediction_error)


# -----------------------------
# Display prediction
# -----------------------------

st.subheader("Prediction")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Predicted RUL",
        f"{prediction:.1f} cycles"
    )

with col2:
    st.metric(
        "Actual RUL",
        f"{actual_rul:.0f} cycles"
    )

with col3:
    st.metric(
        "Absolute Error",
        f"{absolute_error:.1f} cycles"
    )


# -----------------------------
# Interpretation
# -----------------------------

st.subheader("Engine Status")

if prediction <= 50:
    st.error("🔴 Low remaining useful life")
elif prediction <= 100:
    st.warning("🟡 Moderate remaining useful life")
else:
    st.success("🟢 Relatively high remaining useful life")

if prediction_error > 0:
    st.info(
        f"⚠️ The model overestimated RUL by "
        f"{absolute_error:.1f} cycles."
    )
elif prediction_error < 0:
    st.info(
        f"ℹ️ The model underestimated RUL by "
        f"{absolute_error:.1f} cycles."
    )
else:
    st.info("✅ The prediction matches the actual RUL.")

st.subheader("Sensor Trends")

selected_sensors = st.multiselect(
    "Select sensors to visualize",
    ["sensor_11", "sensor_4", "sensor_12", "sensor_9", "sensor_7"],
    default=["sensor_11", "sensor_4"]
)

if selected_sensors:
    chart_data = test_df[
        test_df["unit_id"] == selected_engine
    ][["cycle"] + selected_sensors]

    chart_data = chart_data.set_index("cycle")

    st.line_chart(chart_data)


# -----------------------------
# Sensor information
# -----------------------------

st.subheader("Engine Sensor Data")

st.dataframe(
    X_engine.T.rename(columns={X_engine.index[0]: "Value"}),
    use_container_width=True
)


# -----------------------------
# Model information
# -----------------------------
st.subheader("Model Performance")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("MAE", "18.44 cycles")

with col2:
    st.metric("RMSE", "25.37 cycles")

with col3:
    st.metric("R²", "0.627")
st.subheader("How This Prediction Works")

st.write(
    "The model uses the engine's operating cycle and sensor measurements "
    "to estimate its Remaining Useful Life (RUL). "
    "The prediction is generated using a Gradient Boosting Regressor "
    "trained on the NASA C-MAPSS FD001 dataset."
)

st.info(
    "RUL represents the estimated number of operating cycles remaining "
    "before engine failure."
)

st.divider()

st.caption(
    "Model: Gradient Boosting Regressor | "
    "Dataset: NASA C-MAPSS FD001 | "
    "Task: Remaining Useful Life Regression"
)
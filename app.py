import streamlit as st
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="House Price Prediction", page_icon="🏠")

st.title("🏠 House Price Prediction")
st.write("Enter house details to estimate the price.")

# Sample dataset
data = {
    "Area": [800, 1000, 1200, 1500, 1800, 2000, 2400],
    "Bedrooms": [2, 2, 3, 3, 3, 4, 4],
    "Bathrooms": [1, 2, 2, 2, 3, 3, 3],
    "Price": [2500000, 370000, 490000, 580000,
              629000, 700000, 850000]
}

df = pd.DataFrame(data)

X = df[["Area", "Bedrooms", "Bathrooms"]]
y = df["Price"]

# Train model
model = LinearRegression()
model.fit(X, y)

st.subheader("Enter House Details")

area = st.number_input(
    "Area (square feet)",
    min_value=100,
    max_value=10000,
    value=1200,
    step=100
)

bedrooms = st.number_input(
    "Number of Bedrooms",
    min_value=1,
    max_value=20,
    value=3,
    step=1
)

bathrooms = st.number_input(
    "Number of Bathrooms",
    min_value=1,
    max_value=20,
    value=2,
    step=1
)

if st.button("Predict House Price"):
    input_data = pd.DataFrame(
        [[area, bedrooms, bathrooms]],
        columns=["Area", "Bedrooms", "Bathrooms"]
    )

    prediction = max(0, model.predict(input_data)[0])

    st.success(
        f"Estimated House Price: ₹{prediction:,.0f}"
    )

    st.caption(
        "This is an educational estimate based on a small sample dataset, "
        "not an actual market valuation."
    )

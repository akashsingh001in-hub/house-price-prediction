from flask import Flask, render_template, request
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

app = Flask(__name__)
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
    "Price": [250000, 370000, 490000, 580000,
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

data = {
    "Area": [800, 1000, 1200, 1500, 1800, 2000, 2400],
    "Bedrooms": [2, 2, 3, 3, 3, 4, 4],
    "Bathrooms": [1, 2, 2, 2, 3, 3, 3],
    "Price": [
        250000, 370000, 490000, 580000,
        629000, 700000, 850000
    ]
}


df = pd.DataFrame(data)


X = df[["Area", "Bedrooms", "Bathrooms"]]
y = df["Price"]


# Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.29, random_state=42
)

# Define both models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=100, random_state=42
    )
}

# Evaluate both models
for name, candidate in models.items():
    candidate.fit(X_train, y_train)
    predictions = candidate.predict(X_test)

    print(f"\n{name}")
    print("MAE:", round(mean_absolute_error(y_test, predictions), 2))
    print("RMSE:", round(np.sqrt(mean_squared_error(y_test, predictions)), 2))
    print("R2 Score:", round(r2_score(y_test, predictions), 4))

# Final model for website predictions
model = LinearRegression()
model.fit(X, y)


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        area = float(request.form["area"])
        bedrooms = int(request.form["bedrooms"])
        bathrooms = int(request.form["bathrooms"])

        input_data = pd.DataFrame(
            [[area, bedrooms, bathrooms]],
            columns=["Area", "Bedrooms", "Bathrooms"]
        )

        prediction = max(
            0, round(model.predict(input_data)[0])
        )

    return render_template(
        "index.html",
        prediction=prediction
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

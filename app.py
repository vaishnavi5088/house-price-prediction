from flask import Flask, render_template, request
import pandas as pd
import joblib

# --------------------------
# Create Flask App
# --------------------------
app = Flask(__name__)

# Load model + training columns
model = joblib.load("model.pkl")
cols = joblib.load("columns.pkl")

# --------------------------
# Home Page
# --------------------------
@app.route("/")
def index():
    return render_template("index.html")

# --------------------------
# Prediction Route
# --------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        form_data = request.form.to_dict()

        # Convert numeric fields
        numeric_cols = ["OverallQual", "GrLivArea", "TotalBsmtSF", "GarageCars"]

        for col in numeric_cols:
            value = form_data.get(col)
            if value == "" or value is None:
                form_data[col] = None
            else:
                form_data[col] = float(value)

        # Convert to user input DF
        df_user = pd.DataFrame([form_data])

        # Create empty DF with training columns
        df = pd.DataFrame(columns=cols)

        # Insert values
        for col in df_user.columns:
            if col in df.columns:
                df[col] = df_user[col]

        df = df.where(pd.notnull(df), None)

        # Predict
        price = model.predict(df)[0]
        price = round(price, 2)

        # ------------------------------------
        # FIXED LINE: now sends `prediction`
        # ------------------------------------
        return render_template("index.html", prediction=f"${price}")

    except Exception as e:
        return f"Error: {str(e)}"

# --------------------------
# Run the App
# --------------------------
if __name__ == "__main__":
    app.run(debug=True)

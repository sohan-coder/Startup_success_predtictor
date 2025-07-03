from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

#Load the model
model = pickle.load(open("C:/Users/Sohan/OneDrive/Desktop/python/Machine Learning/Startup_pred/startup_model.pkl", "rb"))

#List of exactly 36 features
features = [
    'age_first_funding_year', 'age_last_funding_year', 'age_first_milestone_year', 'age_last_milestone_year',
    'funding_rounds', 'funding_total_usd', 'milestones', 'is_CA', 'is_NY', 'is_MA', 'is_TX', 'is_otherstate',
    'is_software', 'is_web', 'is_mobile', 'is_enterprise', 'is_advertising', 'is_gamesvideo', 'is_ecommerce',
    'is_biotech', 'is_consulting', 'is_othercategory', 'has_VC', 'has_angel', 'has_roundA', 'has_roundB',
    'has_roundC', 'has_roundD', 'avg_participants', 'is_top500', 'has_RoundABCD', 'has_Investor', 'has_both',
    'invalid_startup', 'age_startup_year', 'tier_relationship'
]

# ✅ Homepage route
@app.route('/')
def home():
    return render_template("index.html")  # This should be your landing page (welcome/home.html)

# ✅ Prediction form route
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            # Extract user inputs from form
            input_values = [float(request.form.get(col, 0)) for col in features]
            input_array = np.array([input_values])

            # Debug: check input shape
            print("Input shape:", input_array.shape)

            # Predict using the model
            prediction = model.predict(input_array)[0]
            label = "Acquired ✅" if prediction == 1 else "Closed ❌"

            return render_template("predict.html", columns=features, prediction=label, request=request)

        except Exception as e:
            return render_template("predict.html", columns=features, prediction=f"Error: {str(e)}", request=request)
    
    # On GET request, render form
    return render_template("predict.html", columns=features)
if __name__ == "__main__":
    app.run(debug=True)
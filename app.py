from flask import Flask, request, render_template_string
import joblib

app = Flask(__name__)

# Load model
model = joblib.load("crop_yield_model.pkl")

# Mappings (adjust if needed)
crop_map = {"Rice": 0, "Wheat": 1, "Maize": 2}
state_map = {"Maharashtra": 5, "Assam": 2, "Bihar": 3}

# HTML UI
html = """
<!DOCTYPE html>
<html>
<head>
<title>Crop Yield Predictor</title>
<style>
body {
    font-family: Arial;
    background: linear-gradient(135deg, #667eea, #764ba2);
    text-align: center;
    padding: 40px;
}
.container {
    background: white;
    padding: 30px;
    border-radius: 15px;
    width: 40%;
    margin: auto;
    box-shadow: 0px 10px 30px rgba(0,0,0,0.3);
}
input, select {
    width: 90%;
    padding: 10px;
    margin: 8px;
    border-radius: 8px;
}
button {
    background: #667eea;
    color: white;
    padding: 12px;
    border-radius: 10px;
    width: 60%;
    cursor: pointer;
}
button:hover {
    background: #5a67d8;
}
.result {
    margin-top: 20px;
    font-size: 20px;
    color: green;
    font-weight: bold;
}
.reset {
    background: red;
    margin-top: 10px;
}
</style>
</head>

<body>
<div class="container">
<h2>🌾 Crop Yield Prediction</h2>

<form method="POST">

<select name="crop">
<option value="Rice">Rice</option>
<option value="Wheat">Wheat</option>
<option value="Maize">Maize</option>
</select>

<select name="state">
<option value="Maharashtra">Maharashtra</option>
<option value="Assam">Assam</option>
<option value="Bihar">Bihar</option>
</select>

<input name="temp" placeholder="Temperature (°C)" required>
<input name="rainfall" placeholder="Rainfall (mm)" required>
<input name="humidity" placeholder="Humidity (%)" required>
<input name="n" placeholder="Nitrogen (N)" required>
<input name="ph" placeholder="pH Value" required>

<button type="submit">Predict Yield</button>
</form>

<button class="reset" onclick="window.location.href='/'">Reset</button>

{% if prediction %}
<div class="result">
🌟 Predicted Yield: {{ prediction }} t/ha <br>
🔍 Confidence: ±{{ confidence }}
</div>
{% endif %}

</div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    confidence = None

    if request.method == "POST":
        try:
            # Dropdown values
            crop = crop_map[request.form["crop"]]
            state = state_map[request.form["state"]]

            # User inputs
            temp = float(request.form["temp"])
            rainfall = float(request.form["rainfall"])
            humidity = float(request.form["humidity"])
            n = float(request.form["n"])
            ph = float(request.form["ph"])

            # Feature Engineering (same as training)
            ndvi = 0.7
            evi = 0.3
            gdd = temp - 10 if temp > 10 else 0
            rainfall_var = rainfall * 0.1

            # Final feature vector (16 features)
            values = [
                crop,
                2020,
                state,
                500000,  # fertilizer
                2000,    # pesticide
                temp,
                rainfall,
                humidity,
                n,
                20,      # P
                40,      # K
                ph,
                ndvi,
                evi,
                gdd,
                rainfall_var
            ]

            pred = model.predict([values])[0]
            prediction = round(pred, 2)
            confidence = round(pred * 0.1, 2)

        except Exception as e:
            prediction = f"Error: {e}"

    return render_template_string(html,
                                  prediction=prediction,
                                  confidence=confidence)

if __name__ == "__main__":
    app.run(debug=True)
# Household Carbon Footprint ML App

This project predicts household carbon footprint levels using machine learning. It classifies households into **low**, **medium**, or **high** emission groups and estimates monthly CO₂ emissions.

Live app:
https://household-carbon-footprint-ml.streamlit.app/

## Project Structure

```text
app/          Streamlit application
data/         Original and processed datasets
models/       Trained model files
notebooks/    Model development and analysis notebooks
tests/        Unit tests
docs/         Manual frontend test plan
```

## Main Functions

* Carbon level prediction: low / medium / high
* Monthly CO₂ estimation
* SHAP-based explanation of important features
* Per-capita comparison
* What-if simulator
* Optional AI advice using Groq API

## Run Locally

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the app:

```bash
streamlit run app/app01_main.py
```

Run tests:

```bash
pytest tests/test_app.py -v
```

## API Key

For local AI advice, create:

```text
.streamlit/secrets.toml
```

and add:

```toml
GROQ_API_KEY = "your_api_key_here"
```

This file should not be uploaded to GitHub. For Streamlit Cloud, the same key is added in the app's **Secrets** settings.

## Note

This app is an academic demonstration. The results are approximate and should not be treated as a professional carbon audit.

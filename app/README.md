# App Folder

This folder contains the Streamlit application.

## Files

```text
app01_main.py      Main Streamlit interface
app02_model.py     Model loading, feature engineering and prediction
app03_advice.py    SHAP explanation and optional AI advice
app04_config.py    Shared settings, colours and rule-based tips
```

## Run the App

Run this command from the project root:

```bash
streamlit run app/app01_main.py
```

The app expects the following folders to exist in the project root:

```text
data/
models/
```

The optional Groq API key should be stored in:

```text
.streamlit/secrets.toml
```

with:

```toml
GROQ_API_KEY = "your_api_key_here"
```

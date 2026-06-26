# Notebooks Folder

This folder contains the data analysis and model development notebooks.

## Notebook Order

```text
01_data_understanding.ipynb
02_data_preprocessing.ipynb
03_logistic_regression.ipynb
04_xgboost.ipynb
05_random_forest.ipynb
06_kmeans.ipynb
07_regression.ipynb
08_logistic_regression_binary.ipynb
09_stacking.ipynb
10_feature_selection_and_pca.ipynb
```

## How to Run

Open the project root folder in VS Code or Jupyter, then run the notebooks in numerical order.

The notebooks use files from:

```text
data/
```

and save trained model files to:

```text
models/
```

## Notes

* `02_data_preprocessing.ipynb` creates the processed dataset.
* Classification models are developed in notebooks 03, 04, 05, 08 and 09.
* `07_regression.ipynb` trains the monthly CO₂ regression model.
* `10_feature_selection_and_pca.ipynb` contains feature selection and PCA analysis.

If running all notebooks from the beginning, make sure the required dataset exists in the `data/` folder first.

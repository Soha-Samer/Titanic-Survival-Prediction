# Titanic Survival Prediction

Predicting which passengers survived the Titanic, using Python and scikit-learn. The project covers data cleaning, feature engineering, statistical checks and a Random Forest classifier.

**Question:** which passenger characteristics are linked to survival, and how well can survival be predicted from them?

## Results

| Metric | Value |
|---|---|
| Hold-out accuracy (80/20 split, 179 passengers) | 83.8% |
| Survivors: precision / recall / F1 (hold-out) | 0.81 / 0.80 / 0.80 |
| 5-fold cross-validation accuracy (mean) | 80.7% (folds range 75.3% to 84.3%) |
| Experiment: PCA reduced to 2 components | 73.7% (full feature set kept) |

The cross-validated score is the more reliable estimate, because the hold-out set is small. The hold-out model used balanced class weights; the cross-validation run used a default Random Forest.

## Data

Kaggle "Titanic: Machine Learning from Disaster" training set: 891 passengers, 12 columns, 38.4% survived (342 of 891).

Missing values: Age 19.9% (177 rows), Cabin 77.1%, Embarked 0.2% (2 rows).

## Approach

1. **Missing values:** Embarked filled with the most common port; Age filled with the average age of the passenger's title group; Cabin turned into a deck feature (no cabin recorded = 0).
2. **Feature engineering:** title extracted from the name with a regular expression (Mr, Mrs, Miss, Rare) and cabin deck extracted from the cabin code.
3. **Encoding:** Sex as 0/1; Embarked, title and deck as integer codes.
4. **Exploration:** histograms, boxplots, chi-square and Cramér's V for a first look at how each feature relates to survival.
5. **Modelling:** Random Forest with balanced class weights to handle the 62/38 class split, plus a PCA experiment.
6. **Evaluation:** hold-out test set and 5-fold cross-validation.

## Limitations and next steps

- Imputation and scaling were done on the full dataset before splitting, which can make scores slightly optimistic. Next step: move preprocessing into a scikit-learn Pipeline fitted on training folds only.
- The notebook includes commented-out experiments (Logistic Regression, SVM, XGBoost, SMOTE, manual class weights). Next step: run them and compare models with cross-validation.
- Chi-square scores depend on feature scale, so they are indicative only. Next step: add Random Forest feature importances.

## Tech

Python, Pandas, NumPy, scikit-learn, SciPy, Matplotlib, Seaborn

## Run it

```
pip install pandas numpy scikit-learn scipy matplotlib seaborn xgboost jupyter
```

Download `train.csv` from the Kaggle competition page, place it next to the notebook, and open `titanic_survival_prediction.ipynb`.

## Author

Soha Elansary, [LinkedIn](https://linkedin.com/in/soha-samer-352504230)

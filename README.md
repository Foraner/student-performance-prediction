# Student Performance Prediction with Custom Sklearn Pipelines

A regression pipeline that predicts Portuguese students' final grades (0-20 scale) to help school advising teams identify at-risk students for early intervention. The key finding: previous term grades are the dominant predictor, meaning the tool becomes useful after the first grading period, not on day one.

## Tools
- Python (Pandas, NumPy)
- Scikit-learn (custom transformers, ColumnTransformer, GridSearchCV, cross-validation)
- Matplotlib / Seaborn (EDA and visualization)

## Data
- Source: [UCI Student Performance Dataset](https://archive.ics.uci.edu/ml/datasets/student+performance)
- 395 students, 35 features (demographics, family background, study habits, interim grades)
- Target: **G3**, the final year grade on a 0-20 scale

## Approach
1. Built a custom transformer class (inheriting from BaseEstimator and TransformerMixin) with a configurable toggle to include or exclude interim grades (G1, G2) and a feature that combines three absence columns into a single total
2. Used ColumnTransformer to apply StandardScaler to numeric features, OneHotEncoder to categorical features, and OrdinalEncoder to ordinal features
3. Compared three models via 3-fold cross-validation: Linear Regression, SVR, and Lasso Regression
4. Tuned SVR hyperparameters (C, epsilon, kernel) using GridSearchCV across 18 combinations
5. Evaluated on a held-out 20% test set

## Results

| Model | RMSE (with G1/G2) |
|-------|-------------------|
| Linear Regression | 2.11 |
| Lasso Regression | 2.11 |
| SVR (default) | 2.60 |
| **SVR (tuned: C=1, ε=0.5, linear)** | **1.97** |

**Final test set: RMSE 2.14, R² = 0.78**

Without interim grades (G1/G2), R² dropped to 0.17. That means demographics and behavior alone are poor predictors. The model only becomes useful after first-term grades are available.

## Key Takeaways
- Custom transformers make pipelines configurable and reusable. The drop_grades toggle enabled running the full pipeline in two configurations without code changes
- Hyperparameter tuning transformed SVR from the weakest model to the strongest
- The biggest insight was practical, not technical: early intervention tools need first-term data to be meaningful

## How to Run
1. Clone this repo
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Open the notebook and run cells top to bottom

---
**Patrick Foran** | [LinkedIn](https://www.linkedin.com/in/patrickmforan) · patrickmforan@gmail.com

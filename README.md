# Data_mining_project
Data mining about heart disease

IT160IU - Data Mining, Fall 2026. A classification framework built from scratch on the
[Heart Disease dataset](https://www.kaggle.com/datasets/oktayrdeki/heart-disease): Decision Tree (J48)
and Random Forest, implemented in Java using the Weka library, evaluated with 10-fold cross-validation.

## Team & Task Assignment

| Member    | Role                          | Step          | Status        |
|-----------|-------------------------------|---------------|---------------|
| Person 1  | Dataset & Pre-processing      | Step 1        | ✅ Done        |
| Person 2  | Algorithm 1 (Decision Tree)   | Step 2        | ⏳ In progress |
| Person 3  | Algorithm 2 & Evaluation      | Step 3 + 4    | ⏳ In progress |

## Dataset

- **Raw file:** `code/dataset/heart_disease.csv` — 10,000 rows, 21 columns (20 attributes + target)
- **Target:** `Heart Disease Status` (`Yes` / `No`) — imbalanced, ~80% No / ~20% Yes
- **9 numeric attributes:** Age, Blood Pressure, Cholesterol Level, BMI, Sleep Hours, Triglyceride Level,
  Fasting Blood Sugar, CRP Level, Homocysteine Level
- **11 categorical attributes:** Gender, Exercise Habits, Smoking, Family Heart Disease, Diabetes,
  High Blood Pressure, Low HDL Cholesterol, High LDL Cholesterol, Alcohol Consumption, Stress Level,
  Sugar Consumption

## Project Structure

```
code/
├── src/
│   ├── Preprocessing.ipynb      ✅ Person 1 (Python - data cleaning)
│   ├── DataAnalysis.java        ✅ Person 1 (Java - data analysis + ARFF export)
│   ├── DecisionTreeJ48.java     ⏳ Person 2
│   ├── RandomForestModel.java   ⏳ Person 3
│   └── Evaluation.java          ⏳ Person 3
├── lib/
│   └── weka.jar
├── dataset/
│   ├── heart_disease.csv            (raw)
│   ├── cleaned_heart_disease.csv    (output of Preprocessing.ipynb)
│   ├── train.arff                   (output of DataAnalysis.java)
│   ├── test.arff                    (output of DataAnalysis.java)
│   └── evaluation_results.csv       (output of Evaluation.java)
└── models/
    ├── DECISIONTREE.model
    └── RANDOMFOREST.model
```

---

## Step 1 — Dataset & Pre-processing (Person 1) ✅

Per the instructor's guidance, this step is split across two languages: **Python is used only for
data cleaning and to export a clean CSV; all data analysis and the ARFF export are done in Java/Weka.**

### 1. `Preprocessing.ipynb` (Python) — Data Cleaning

Reads the raw CSV and performs:
- Duplicate check (0 found)
- Missing value handling:
  - `Alcohol Consumption` (~25.9% missing) → recoded as its own category, `"Unknown"`
  - All other columns (<0.3% missing each) → affected rows dropped (468 rows, 4.68% of data;
    class ratio unchanged, ≈80/20 before and after)
- Exports `code/dataset/cleaned_heart_disease.csv` (9,532 rows × 21 columns, 0 missing values)

**Run:** open in Jupyter / VS Code and Run All. Default paths assume the notebook sits in `code/src/`
and reads/writes relative to `code/dataset/`.

### 2. `DataAnalysis.java` (Java) — Data Analysis + ARFF Export

Reads `cleaned_heart_disease.csv` and, entirely in Java/Weka, performs:
- Target variable distribution
- Outlier detection (IQR rule) on all 9 numeric attributes
- Multicollinearity check among numeric attributes (Pearson correlation)
- Numeric attributes vs. target: correlation + t-test
- Categorical attributes vs. target: chi-square test of independence
- Stratified 80/20 train/test split (fixed seed, reproducible)
- ARFF export: `train.arff`, `test.arff`

**Key findings:**
- 0 outliers in any numeric attribute; max pairwise correlation between numeric attributes = 0.023
  (no multicollinearity)
- Only `BMI` is statistically significant vs. the target (r = 0.021, p = 0.039) — significant but
  practically negligible (very small correlation, large sample size n = 9,532)
- No categorical attribute is significantly associated with the target (all p > 0.3)
- **Conclusion:** no single attribute is individually predictive of heart disease in this dataset —
  relevant for interpreting Step 2-4 results (a model that only reaches ~80% accuracy may just be
  predicting the majority class; Precision/Recall/F1 of the "Yes" class should be reported, not
  Accuracy alone)

**Compile & run** (from `code/src/`):
```
javac -cp "../lib/weka.jar" -d . DataAnalysis.java
java --add-opens java.base/java.lang=ALL-UNNAMED -cp "../lib/weka.jar;." DataAnalysis
```
(On Linux/Mac, replace `;` with `:` in the classpath.)

### Requirements

- **Java JDK 8+** and **`weka.jar`** (place at `code/lib/weka.jar`) — needed for `DataAnalysis.java`
  and every other `.java` file in this project.
- **Python 3 + pandas** — needed only for `Preprocessing.ipynb`.

---

## Steps 2-4 (Person 2 & Person 3)

To be filled in once Person 2 (`DecisionTreeJ48.java`) and Person 3 (`RandomForestModel.java`,
`Evaluation.java`) finish their parts.

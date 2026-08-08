# Exploratory Data Analysis (EDA) & Machine Learning Preprocessing Projects

[![Python Version](https://img.shields.io/badge/Python-3.8+-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0+-darkblue.svg?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.20+-blue.svg?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.2+-orange.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-0.12+-blueviolet.svg?style=for-the-badge)](https://seaborn.pydata.org/)

Welcome to this repository containing a collection of advanced **Exploratory Data Analysis (EDA)**, **Data Wrangling**, **Feature Engineering**, and **Statistical Feature Selection** projects implemented in Jupyter Notebooks. These projects showcase end-to-end data pipelines from loading raw datasets to preparing clean, encoded, and scaled features suitable for machine learning models.

---

## 📂 Projects Directory

| Project Name | Notebook Link | Core Focus / Highlights | Key Techniques |
| :--- | :--- | :--- | :--- |
| **Amazon Prime Movies EDA** | [Amazon_Prime_EDA.ipynb](file:///c:/Users/kotir/EDA_projects/Amazon_Prime_EDA.ipynb) | Catalog Content analysis | Data wrangling, duplication cleaning, metadata extraction, rating analysis. |
| **Heart Failure Prediction EDA** | [Eda & datacleaning _heart.csv.ipynb](file:///c:/Users/kotir/EDA_projects/Eda%20&%20datacleaning%20_heart.csv.ipynb) | Clinical feature extraction & reduction | Outlier treatment, Standard Scaling, Age grouping, Dimensionality reduction via **PCA**. |
| **Medical Insurance Cost EDA** | [Eda & datacleaning _insurance.csv.ipynb](file:///c:/Users/kotir/EDA_projects/Eda%20&%20datacleaning%20_insurance.csv.ipynb) | Biometric/lifestyle charge patterns | One-hot encoding, BMI binning, **Pearson Correlation** & **Chi-Square** feature selection. |
| **Hotel Booking Analysis (EDA)** | [Hotel_Booking_Analysis(EDA).ipynb](file:///c:/Users/kotir/EDA_projects/Hotel_Booking_Analysis%28EDA%29.ipynb) | Business performance optimization | Cancellation analysis, ADR trends, customer demographics, seasonality profiling. |

---

## 🔍 Detailed Project Profiles

### 🎬 1. Amazon Prime Movies EDA
*   **Overview**: An exploratory analysis of Amazon Prime's movie catalog to identify patterns in maturity ratings, user interest via IMDb scores, and runtime lengths.
*   **Target Variables & Metadata**: `Movie Name`, `Language`, `IMDb Rating`, `Running Time`, `Year of Release`, `Maturity Rating`, `Plot`.
*   **Data Wrangling & Cleaning**:
    *   Identified and removed duplicate movie records to avoid statistical skewing.
    *   Standardized text formats and extracted numeric ratings.
*   **Key Findings**:
    *   Determined the most prevalent maturity ratings (e.g., `18+`, `All`, `13+`) and their relative counts in the catalog.
    *   Analyzed how content additions and release years evolved over the decades, highlighting the platform's rapid expansion.
    *   Investigated top-rated languages on Prime, mapping their corresponding average IMDb scores.

### 💓 2. Heart Failure Prediction (EDA & Feature Engineering)
*   **Overview**: Exploring cardiovascular clinical metrics to structure model-ready datasets for predicting heart disease incidents.
*   **Clinical Features**: `Age`, `Sex`, `ChestPainType`, `RestingBP`, `Cholesterol`, `FastingBS`, `RestingECG`, `MaxHR`, `ExerciseAngina`, `Oldpeak`, `ST_Slope`, `HeartDisease` (Target).
*   **Preprocessing Pipeline**:
    *   **Outlier Treatment**: Corrected anomalies in `RestingBP` and `Cholesterol` fields (e.g., handling records with zero cholesterol through statistical cleaning).
    *   **Feature Scaling**: Standardized numerical features (`Age`, `RestingBP`, `Cholesterol`, `MaxHR`, `Oldpeak`) using `StandardScaler` to bring them to a uniform variance.
    *   **Categorization**: Grouped continuous `Age` values into categorical bins (`Young`, `Middle`, `Senior`) and applied dummy variable encoding.
*   **Advanced Feature Reduction**:
    *   **PCA (Principal Component Analysis)**: Reduced one-hot encoded dataset features from **15 down to 11 principal components** while preserving **96.7%** of the total explained variance. This ensures reduced overfitting and faster training for downstream ML estimators.

### 🏥 3. Medical Insurance Cost (EDA & Preprocessing)
*   **Overview**: Exploring the effects of demographic traits (age, region, children) and lifestyle choices (smoking, obesity) on medical insurance billing charges.
*   **Dataset Schema**: `age`, `sex`, `bmi`, `children`, `smoker`, `region`, `charges` (Target).
*   **Preprocessing & Feature Engineering**:
    *   **Feature Scaling**: Continuous inputs (`age`, `bmi`, `children`) scaled via `StandardScaler`.
    *   **Categorical Encoding**: Mapped binary columns (`sex` $\rightarrow$ `is_female`, `smoker` $\rightarrow$ `is_smoker`) and dummy-encoded regional categories.
    *   **BMI Binning**: Engineered a `bmi_category` parameter dividing values into `Underweight`, `Normal`, `Overweight`, and `Obese`.
*   **Statistical Feature Selection**:
    *   **Pearson Correlation Coefficient**: Evaluated continuous features against `charges`. Found high direct positive correlation with `age` and `bmi`.
    *   **Chi-Square Contingency Test**: Partitioned charges into quantiles and ran Chi-Square independence tests on categorical variables. Features failing to reject the null hypothesis at $\alpha = 0.05$ (e.g., `region_southwest`, `bmi_category_Normal`, `bmi_category_Overweight`, `region_northwest`) were systematically dropped, retaining only highly significant features (`is_smoker`, `region_southeast`, `is_female`, `bmi_category_Obese`) for clean modeling.

### 🏨 4. Hotel Booking Analysis (EDA)
*   **Overview**: A comprehensive operational analysis comparing resort and city hotel bookings to optimize occupancy, revenue management (ADR), and cancel rates.
*   **Core Objectives**:
    *   **Cancellation Profiling**: Evaluated cancellation patterns as a function of lead times, deposit configurations (`No Deposit` vs. `Non Refund`), customer types, and distribution channels.
    *   **Seasonality & Demand**: Plotted monthly occupancy patterns and mapped them alongside the Average Daily Rate (ADR) to locate optimal pricing and demand peak seasons.
    *   **Customer Segmentation**: Studied demographic split (singles, couples, families with kids/babies) and repeat booking behaviors to optimize targeting strategies.
    *   **Operational Bottlenecks**: Analyzed parking request trends, booking adjustments, and special request counts to help streamline hotel resource allocation.

---

## 🛠️ Environment Setup & How to Run

Follow these instructions to run the notebooks on your local system:

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2. Set Up a Virtual Environment (Recommended)
Using Python's built-in `venv`:
```bash
python -m venv venv
# On Windows (PowerShell/CMD)
.\venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies
Install all the required data science and plotting libraries:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy jupyter
```

### 4. Run Jupyter Notebook
Launch the Jupyter interface and select the project notebook you wish to run:
```bash
jupyter notebook
```

---

## 📊 Summary of Technical Stack

*   **Data Wrangling & Processing**: Pandas, NumPy
*   **Visualization & EDA**: Matplotlib, Seaborn
*   **Machine Learning Preprocessing**: Scikit-Learn (`StandardScaler`, `PCA`, `OneHotEncoder`)
*   **Statistical Analysis**: SciPy (`pearsonr`, `chi2_contingency`)

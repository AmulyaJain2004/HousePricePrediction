# House Price Prediction Analysis

A comprehensive data science project analyzing real estate pricing factors and building predictive models for house prices per unit area.

## 📊 Project Overview

This project performs an in-depth analysis of real estate data to understand the factors that influence house prices and develop machine learning models for price prediction. The analysis covers data preprocessing, exploratory data analysis, feature engineering, and model development using various regression techniques.

## 🏠 Dataset Description

The project uses a real estate dataset (`Real estate.csv`) containing the following features:

### Features (Independent Variables):

- **X1 - Transaction Date**: The year and time when the property was sold
- **X2 - House Age**: Age of the house in years
- **X3 - Distance to Nearest MRT Station**: Distance to public transportation (in meters)
- **X4 - Number of Convenience Stores**: Count of nearby convenience stores
- **X5 - Latitude**: Geographic coordinate (latitude)
- **X6 - Longitude**: Geographic coordinate (longitude)

### Target Variable:

- **Y - House Price of Unit Area**: The price per unit area (target for prediction)

## 🔍 Analysis Components

### 1. Dataset Overview & Data Quality Assessment

- Dataset dimensions and structure analysis
- Data type identification and conversion
- Missing value detection
- Duplicate record identification
- Date format standardization (fractional year to datetime conversion)

### 2. Statistical Summary & Distribution Analysis

- Comprehensive statistical metrics (mean, median, std, min, max, quartiles)
- Distribution analysis and skewness detection
- Outlier identification using IQR method
- Data normalization techniques (square root transformation)

### 3. Data Cleaning & Preprocessing

- Missing value handling
- Duplicate removal
- Outlier removal using statistical methods
- Feature engineering (extracting year and month from transaction dates)
- Age categorization (0-10, 10-20, 20-30, 30-40, >40 years)

### 4. Exploratory Data Analysis (EDA)

- **Univariate Analysis**: Distribution plots and histograms
- **Feature Exploration**: Box plots for outlier detection
- **Temporal Analysis**:
  - Month-wise and year-wise sales trends
  - Seasonal patterns in real estate transactions
- **Relationship Analysis**: Scatter plots between features and target variable
- **Categorical Analysis**: House age categories and convenience store distributions

### 5. Feature Engineering & Correlation Analysis

- One-hot encoding for categorical variables
- Correlation matrix generation and visualization
- Feature importance ranking
- Heatmap visualization for feature relationships

### 6. Machine Learning Model Development

- **Data Preparation**: Train-test split (80-20)
- **Feature Scaling**: StandardScaler for normalization
- **Model Selection**: Ridge regression with hyperparameter tuning
- **Model Evaluation**: GridSearchCV for optimal parameter selection
- **Performance Metrics**: MSE, R² score, and cross-validation

## 📈 Key Insights

### Statistical Findings:

- Average house price: 37.98 per unit area
- Price range: 7.6 to 117.5 (significant variation)
- Average house age: 17.71 years
- Average distance to MRT: 1,083.89 meters
- Average convenience stores nearby: 4.09

### Distribution Characteristics:

- House prices show moderate positive skewness (0.5977)
- Square root transformation effectively normalizes the distribution
- Most properties are within 1,454 meters of MRT stations (75th percentile)

### Temporal Patterns:

- Data spans from September 2012 to August 2013
- Clear seasonal variations in real estate transactions
- Different activity patterns between 2012 and 2013

## 🛠️ Technologies Used

### Python Libraries:

- **Data Manipulation**: `pandas`, `numpy`
- **Visualization**: `matplotlib`, `seaborn`
- **Statistical Analysis**: `scipy.stats`
- **Machine Learning**: `scikit-learn`
  - Model Selection: `train_test_split`, `GridSearchCV`
  - Preprocessing: `StandardScaler`, `OneHotEncoder`
  - Models: `Ridge`, `LinearRegression`
  - Metrics: `mean_squared_error`, `r2_score`

## 📁 Project Structure

```
HousePricePrediction/
├── HousePricePrediction.ipynb    # Main analysis notebook
├── Real estate.csv               # Dataset
└── README.md                     # Project documentation
```

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn
```

### Running the Analysis

1. Clone the repository
2. Ensure all required libraries are installed
3. Open `HousePricePrediction.ipynb` in Jupyter Notebook or VS Code
4. Run cells sequentially to reproduce the analysis

## 📊 Model Performance

The project implements Ridge regression with hyperparameter optimization:

- **Cross-validation**: 5-fold CV for robust evaluation
- **Hyperparameter tuning**: Alpha values tested: [0.1, 1.0, 10.0, 100.0]
- **Evaluation metrics**: MSE and R² score for model assessment

## 🎯 Business Applications

This analysis provides valuable insights for:

- **Real Estate Investors**: Understanding price drivers and market trends
- **Property Developers**: Location selection and pricing strategies
- **Urban Planners**: Impact of infrastructure on property values
- **Financial Institutions**: Property valuation and risk assessment

## 📝 Future Enhancements

- Advanced feature engineering (interaction terms, polynomial features)
- Additional machine learning algorithms (Random Forest, Gradient Boosting)
- Time series analysis for price forecasting
- Geographic clustering analysis
- Integration of external economic indicators

---

_This project demonstrates comprehensive data science workflow from data exploration to model development, providing actionable insights for real estate market analysis._

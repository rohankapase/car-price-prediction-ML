# Car Price Prediction using Machine Learning

### Project Overview

This project aims to predict the resale value of used cars using **Regression** techniques. Since car prices depend on complex factors like brand reputation, mileage, fuel type, and age, this machine learning model provides a data-driven approach to estimate fair market value.

### Key Features

* **Data Cleaning:** Handling missing values and removing irrelevant columns.
* **Feature Engineering:** Deriving the "Age" of the car to better reflect depreciation.
* **Categorical Encoding:** Converting text data (Fuel Type, Transmission) into numerical format using One-Hot Encoding.
* **Predictive Modeling:** Using **Random Forest Regressor** for high-accuracy price estimation.

### Tech Stack

* **Language:** Python 3.x
* **Libraries:** * `Pandas` (Data Manipulation)
* `Scikit-Learn` (Machine Learning)
* `Matplotlib` & `Seaborn` (Data Visualization)



### Machine Learning Pipeline

1. **Exploratory Data Analysis (EDA):** Visualizing correlations between features like mileage and price.
2. **Preprocessing:** Encoding categorical variables and splitting the dataset.
3. **Training:** Fitting the Random Forest algorithm on 80% of the data.
4. **Evaluation:** Testing the model performance using the R-squared (R2) score.

### Results

The model successfully identifies that **Present Price** and **Vehicle Age** are the most significant factors in determining the resale value.

### How to Run

1. Clone the repository:
```bash
git clone https://github.com/rohankapase/car-price-prediction-ML.git

```


2. Install dependencies:
```bash
pip install pandas scikit-learn matplotlib seaborn

```


3. Run the script:
```bash
python car_price_model.py

```



### Dataset

The dataset used (`car data.csv`) includes features such as:

* Selling Price (Target)
* Present Showroom Price
* Kilometers Driven
* Fuel Type (Petrol/Diesel/CNG)
* Transmission (Manual/Automatic)
* Year of Purchase

Here is a concise and professional **Conclusion** for your project. You can add this to your project report, presentation, or the end of your README file.

---

### **Conclusion**

The **Car Price Prediction** project successfully demonstrates how machine learning can be used to determine the market value of used vehicles. By implementing the **Random Forest Regressor**, the model effectively captured the complex relationships between a car's features and its resale price.

**Key Takeaways:**

* **Primary Value Drivers:** The analysis revealed that the **Present Showroom Price** and **Vehicle Age** are the most critical factors influencing resale value.
* **Model Performance:** The model achieved a high **R-squared score**, indicating that it can explain most of the variance in car prices and provide reliable estimates.
* **Data-Driven Decisions:** This tool eliminates guesswork for buyers and sellers, providing a transparent, evidence-based valuation instead of relying on intuition.

Overall, this project showcases the power of **predictive analytics** in the automotive industry, providing a scalable solution for real-time price forecasting.

---

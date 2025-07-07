#  E-commerce Sales Forecasting

This repository contains a Python-based project for forecasting sales using the **Pakistan Largest Ecommerce Dataset**. The model utilizes Facebook's **Prophet** library to predict future sales trends, focusing specifically on the **"Mobiles & Tablets"** product category.

---

##  Overview

- **Dataset:** Pakistan Largest Ecommerce Dataset  
- **Tools & Libraries:** Python, Prophet, Pandas, Matplotlib, Seaborn, Scikit-learn  
- **Goal:** Predict sales trends and provide actionable business insights.

---

##  Features

-  Data cleaning & preprocessing (missing values, outliers, type conversions)
-  Exploratory Data Analysis (EDA) on:
  - Grand totals by category
  - Cancellation vs. completion rates
  - Normal vs. discounted purchases
-  Time series forecasting using Prophet:
  - Includes holiday effects (Eid, Ramadan, Independence Day)
  - Adds custom regressors (discounts, lagged sales)
-  Visualization of forecasts, trends, and comparison with actual sales

---

##  Results

- **Selected Category:** Mobiles & Tablets  
- **Total Sales:** ₨ 1.94564 Billion  
- **Model Performance:**  
  - MAPE: **86.08%**  
  - RMSE: **₨ 112,034,103.25**  
- **Predicted Peak Sales Month:** December 2017  
- **Business Insight:**  
  - Increase inventory & marketing during peak months  
  - High sales volatility (CV > 0.5) indicates need for finer-grained modeling

---

##  Project Files

| File | Description |
|------|-------------|
| `E-commerce_model.py` | Main script for data cleaning, EDA, and Prophet forecasting |
| `forecast_Mobiles & Tablets.csv` | Forecasted sales values for the selected category |
| `forecast_plot_Mobiles & Tablets.png` | Sales forecast visualization |
| `forecast_components_Mobiles & Tablets.png` | Prophet components: trend, weekly/yearly seasonality, holidays |
| `actual_vs_predicted_Mobiles & Tablets.png` | Plot comparing actual vs. predicted sales values |
| `Code 1.png` | Code Screenshot |
| `Code 2.png` | Code Screenshot |
| `Output1.png` | Output Screenshot |
| `Output12.png` | Output Screenshot |
---


##  Contact

For questions or collaboration, feel free to connect via GitHub Issues or email.

---

 *If you find this project helpful, feel free to star the repository!*

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
from prophet import Prophet
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error


try:
    from prophet.diagnostics import cross_validation
    CROSS_VALIDATION_AVAILABLE = True
except ImportError:
    print("Warning: prophet.diagnostics.cross_validation not found. Skipping cross-validation.")
    CROSS_VALIDATION_AVAILABLE = False


df = pd.read_csv("Pakistan Largest Ecommerce Dataset.csv", encoding="utf-8", encoding_errors="ignore", low_memory=False)

# Data Preprocessing
print("--- Initial Data Shape:", df.shape, "---")
df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
df['discount_amount'] = pd.to_numeric(df['discount_amount'], errors='coerce').fillna(0)
df['grand_total'] = pd.to_numeric(df['grand_total'], errors='coerce').fillna(0)
df['qty_ordered'] = pd.to_numeric(df['qty_ordered'], errors='coerce').fillna(0)
df['category_name_1'] = df['category_name_1'].fillna('Unknown')
df['payment_method'] = df['payment_method'].fillna('Unknown')

# Handle outliers in grand_total (cap at 99th percentile)
grand_total_cap = df['grand_total'].quantile(0.99)
df['grand_total'] = df['grand_total'].clip(upper=grand_total_cap)

# Derived Features
df['month_year'] = df['created_at'].dt.to_period('M').dt.to_timestamp('M')  # Month-end dates
df['is_cancelled'] = df['status'].str.lower().eq('cancelled').astype(int)
df['is_discounted'] = (df['discount_amount'] > 0).astype(int)

# Save cleaned file
df.to_csv("cleaned_ecommerce_data.csv", index=False)
print("--- Cleaned data saved to 'cleaned_ecommerce_data.csv'.---")

# EDA 1: Grand Total by Top 10 Categories
top_categories = df['category_name_1'].value_counts().nlargest(10).index.tolist()
grand_total_by_top_cat = df[df['category_name_1'].isin(top_categories)].groupby('category_name_1')['grand_total'].sum()
print("\n--- Grand Total by Top Categories:\n", grand_total_by_top_cat, "\n---")

# EDA 2: Sales of All Categories Over Time
sales_over_time = df.groupby(['month_year', 'category_name_1'])['grand_total'].sum().reset_index()
print("\n--- Sales Over Time (First 5 Rows):\n", sales_over_time.head(), "\n---")

# EDA 3: Cancellation Rate vs Completed Rate
cancel_stats = df.groupby('category_name_1')['is_cancelled'].agg(['sum', 'count'])
cancel_stats = cancel_stats.assign(completed=cancel_stats['count'] - cancel_stats['sum'])
cancel_stats.columns = ['cancelled', 'total_orders', 'completed']
print("\n--- Cancellation vs Completion (First 5 Rows):\n", cancel_stats.head(), "\n---")

# EDA 4: Purchase Volume - Discounted vs Non-Discounted
purchase_volume = df.groupby('is_discounted')['qty_ordered'].sum()
purchase_volume.index = purchase_volume.index.map({0: 'No Discount', 1: 'Discounted'})
print("\n--- Purchase Volume (Discounted vs Not):\n", purchase_volume, "\n---")

# --- Sales Forecasting ---

# Aggregate data by month_year and category_name_1
ts_data = df.groupby(['month_year', 'category_name_1'])['grand_total'].sum().reset_index()
ts_data['month_year'] = pd.to_datetime(ts_data['month_year'])

# Select the best category based on data points, sales volume, and stability
category_counts = ts_data.groupby('category_name_1')['month_year'].count()
category_sales = ts_data.groupby('category_name_1')['grand_total'].sum()
category_std = ts_data.groupby('category_name_1')['grand_total'].std()
category_mean = ts_data.groupby('category_name_1')['grand_total'].mean()
category_metrics = pd.DataFrame({
    'months': category_counts,
    'total_sales': category_sales,
    'std_sales': category_std,
    'mean_sales': category_mean
})
category_metrics['cv'] = category_metrics['std_sales'] / category_metrics['mean_sales']
category_metrics = category_metrics[category_metrics['months'] >= 24].sort_values('total_sales', ascending=False)

if category_metrics.empty:
    print("\n ERROR: No category has 24+ months of data for forecasting!")
    exit()

print("\n--- Top 5 Categories for Forecasting (≥24 months) ---\n")
print(category_metrics[['months', 'total_sales', 'cv']].head(), "\n---")

category = category_metrics['total_sales'].idxmax()
print(f"\n--- Selected Category for Forecasting: {category} ---\n")
print(f"Months of data: {category_metrics.loc[category, 'months']}")
print(f"Total sales: {category_metrics.loc[category, 'total_sales']:.2f} PKR")
print(f"Coefficient of Variation: {category_metrics.loc[category, 'cv']:.4f}")

ts_category = ts_data.loc[ts_data['category_name_1'] == category, ['month_year', 'grand_total']].copy()
ts_category.columns = ['ds', 'y']
ts_category['ds'] = ts_category['ds'].dt.to_period('M').dt.to_timestamp('M')
ts_category = ts_category.groupby('ds')['y'].sum().reset_index()
date_range = pd.date_range(start=ts_category['ds'].min(), end=ts_category['ds'].max(), freq='M')
ts_category = ts_category.set_index('ds').reindex(date_range, fill_value=0).reset_index()
ts_category.columns = ['ds', 'y']
ts_category['y'] = ts_category['y'].interpolate(method='linear').fillna(0)

print("\n--- Data for Forecasting (First 5 Rows) ---\n", ts_category.head())
print("\n--- Number of Data Points ---\n", len(ts_category))
if len(ts_category) < 24:
    print(f"\n ERROR: Insufficient data points ({len(ts_category)} months) for {category}. Need at least 24 months.")
    exit()

train = ts_category.iloc[:-12]
test = ts_category.iloc[-12:]
print("\n--- Train Data Size ---\n", len(train))
print("\n--- Test Data Size ---\n", len(test))
print("\n--- Train Date Range ---\n", train['ds'].min(), "to", train['ds'].max())
print("\n--- Test Date Range ---\n", test['ds'].min(), "to", test['ds'].max())

ts_category['is_discounted'] = df[df['category_name_1'] == category].groupby('month_year')['is_discounted'].mean().reindex(ts_category['ds'], method='ffill').fillna(0)
ts_category['lag_1'] = ts_category['y'].shift(1).fillna(ts_category['y'].mean())

holidays = pd.DataFrame({
    'holiday': 'Eid',
    'ds': pd.to_datetime(['2016-07-31', '2017-06-30', '2018-06-30', '2019-06-30', '2020-07-31']),
    'lower_window': -7,
    'upper_window': 7
})
holidays = pd.concat([holidays, pd.DataFrame({
    'holiday': 'Ramadan',
    'ds': pd.to_datetime(['2016-06-30', '2017-05-31', '2018-05-31', '2019-05-31', '2020-05-31']),
    'lower_window': -30,
    'upper_window': 0
})])
holidays = pd.concat([holidays, pd.DataFrame({
    'holiday': 'Independence Day',
    'ds': pd.to_datetime(['2016-08-14', '2017-08-14', '2018-08-14', '2019-08-14', '2020-08-14']),
    'lower_window': -1,
    'upper_window': 1
})])

model = Prophet(
    holidays=holidays,
    yearly_seasonality='auto',
    changepoint_prior_scale=0.05,
    seasonality_prior_scale=10.0
)
model.add_regressor('is_discounted')
model.add_regressor('lag_1')

train_with_regressors = train.merge(ts_category[['ds', 'is_discounted', 'lag_1']], on='ds', how='left')
train_with_regressors['is_discounted'] = train_with_regressors['is_discounted'].ffill().fillna(0)
train_with_regressors['lag_1'] = train_with_regressors['lag_1'].ffill().fillna(train_with_regressors['lag_1'].mean())
model.fit(train_with_regressors)

future = model.make_future_dataframe(periods=12, freq='M')
future = future.merge(ts_category[['ds', 'is_discounted', 'lag_1']], on='ds', how='left')
future['is_discounted'] = future['is_discounted'].ffill().fillna(0)
future['lag_1'] = future['lag_1'].ffill().fillna(ts_category['y'].mean())
forecast = model.predict(future)

test_with_regressors = test.merge(ts_category[['ds', 'is_discounted', 'lag_1']], on='ds', how='left')
test_predictions = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].merge(test, on='ds', how='inner')

if len(test_predictions) > 0:
    mape = mean_absolute_percentage_error(test_predictions['y'], test_predictions['yhat'])
    rmse = mean_squared_error(test_predictions['y'], test_predictions['yhat'], squared=False)
    within_10pct = (abs(test_predictions['y'] - test_predictions['yhat']) / test_predictions['y'] <= 0.1).mean() * 100
    print(f"\n--- Model Accuracy for {category} ---")
    print(f"MAPE: {mape:.4f} ({mape*100:.2f}%)")
    print(f"RMSE: {rmse:.2f} PKR")
    print(f"Percentage of predictions within 10% of actual: {within_10pct:.2f}%")
else:
    print(f"\n--- ERROR: No matching predictions found for test period! ---")
    exit()

if CROSS_VALIDATION_AVAILABLE:
    try:
        cv_results = cross_validation(model, initial='270 days', period='30 days', horizon='30 days')
        if 'mape' in cv_results.columns:
            cv_mape = cv_results['mape'].mean()
            print(f"\n--- Cross-Validation MAPE for {category}: {cv_mape:.4f} ({cv_mape*100:.2f}%) ---")
    except Exception as e:
        print(f"\n--- Cross-Validation Failed: {str(e)} ---")
else:
    cv_train = ts_category.iloc[:-24]
    cv_test = ts_category.iloc[-24:-12]
    if len(cv_train) >= 12 and len(cv_test) >= 12:
        cv_model = Prophet(holidays=holidays, yearly_seasonality='auto', changepoint_prior_scale=0.05, seasonality_prior_scale=10.0)
        cv_model.add_regressor('is_discounted')
        cv_model.add_regressor('lag_1')
        cv_train_with_regressors = cv_train.merge(ts_category[['ds', 'is_discounted', 'lag_1']], on='ds', how='left')
        cv_train_with_regressors['is_discounted'] = cv_train_with_regressors['is_discounted'].ffill().fillna(0)
        cv_train_with_regressors['lag_1'] = cv_train_with_regressors['lag_1'].ffill().fillna(cv_train_with_regressors['lag_1'].mean())
        cv_model.fit(cv_train_with_regressors)
        cv_future = cv_model.make_future_dataframe(periods=12, freq='M')
        cv_future = cv_future.merge(ts_category[['ds', 'is_discounted', 'lag_1']], on='ds', how='left')
        cv_future['is_discounted'] = cv_future['is_discounted'].ffill().fillna(0)
        cv_future['lag_1'] = cv_future['lag_1'].ffill().fillna(ts_category['y'].mean())
        cv_forecast = cv_model.predict(cv_future)
        cv_predictions = cv_forecast[['ds', 'yhat']].merge(cv_test, on='ds', how='inner')
        if len(cv_predictions) > 0:
            cv_mape = mean_absolute_percentage_error(cv_predictions['y'], cv_predictions['yhat'])
            print(f"\n--- Fallback Cross-Validation MAPE for {category}: {cv_mape:.4f} ({cv_mape*100:.2f}%) ---")

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv(f"forecast_{category}.csv", index=False)
print(f"\n--- Forecast saved to 'forecast_{category}.csv' ---")

plt.figure(figsize=(12, 6))
plt.plot(forecast['ds'], forecast['yhat'], label='Forecast', color='red')
plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='red', alpha=0.1)
plt.plot(test['ds'], test['y'], 'o-', label='Actual (Test)', color='blue')
plt.xlabel('Date')
plt.ylabel('Grand Total (PKR)')
plt.title(f'Sales Forecast for {category}')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"forecast_plot_{category}.png")
plt.show()

model.plot_components(forecast)
plt.savefig(f"forecast_components_{category}.png")
plt.show()

if len(test_predictions) > 0:
    plt.figure(figsize=(12, 6))
    plt.plot(test_predictions['ds'], test_predictions['y'], 'o-', label='Actual Sales', color='blue')
    plt.plot(test_predictions['ds'], test_predictions['yhat'], 's-', label='Predicted Sales', color='red')
    plt.fill_between(test_predictions['ds'], test_predictions['yhat_lower'], test_predictions['yhat_upper'], color='red', alpha=0.1)
    plt.xlabel('Date')
    plt.ylabel('Grand Total (PKR)')
    plt.title(f'Actual vs. Predicted Sales for {category}')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"actual_vs_predicted_{category}.png")
    plt.show()

peak_months = forecast[forecast['yhat'] == forecast['yhat'].max()]['ds'].dt.strftime('%Y-%m')
print(f"\n--- Actionable Insights for {category} ---")
print(f"Predicted peak sales month: {peak_months.values[0]}")
print("Recommendation: Increase inventory and marketing for peak months.")
if category_metrics.loc[category, 'cv'] > 0.5:
    print("Warning: High sales volatility (CV > 0.5). Consider additional regressors or shorter forecast horizons.")

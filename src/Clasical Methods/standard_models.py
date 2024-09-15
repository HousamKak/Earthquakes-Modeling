# standard_models.py

"""
Earthquake Inter-Event Time Modeling - Standard Probability Distributions

This script implements the following steps:
1. Load and prepare earthquake data.
2. Split data into training and testing sets.
3. Fit standard probability distributions using Maximum Likelihood Estimation (MLE):
   - Exponential Distribution
   - Weibull Distribution
   - Gamma Distribution
   - Log-Normal Distribution
4. Perform Goodness-of-Fit tests to assess model fit using functions from statistical_tests.py
5. Generate predictions for the test data.
6. Evaluate predictions using error metrics.
7. Visualize actual vs. predicted inter-event times.
"""

# -----------------------------------
# Import Necessary Libraries
# -----------------------------------

import pandas as pd                # For data manipulation
import numpy as np                 # For numerical computations
from scipy import stats            # For statistical distributions
import matplotlib.pyplot as plt    # For plotting
from sklearn.metrics import mean_absolute_error, mean_squared_error  # For evaluation metrics

# Import statistical test functions
from statistical_tests import ks_test, ad_test, chi_square_test, calculate_aic, calculate_bic

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

# -----------------------------------
# Step 1: Load and Prepare the Data
# -----------------------------------

# Load earthquake data from a CSV file
df = pd.read_csv('./data/earthquake_data.csv')

# Convert the 'time' column to datetime objects
df['time'] = pd.to_datetime(df['time'])

# Rename 'mag' column to 'magnitude' if necessary
if 'magnitude' not in df.columns and 'mag' in df.columns:
    df.rename(columns={'mag': 'magnitude'}, inplace=True)

# Sort the DataFrame by time to ensure chronological order
df = df.sort_values('time')

# Calculate inter-event times in hours
df['inter_event_time'] = df['time'].diff().dt.total_seconds() / 3600

# Remove the first NaN value resulting from the diff()
inter_event_times = df['inter_event_time'].dropna().reset_index(drop=True)

# -----------------------------------
# Step 2: Split Data into Training and Testing Sets
# -----------------------------------

# Determine the split index (e.g., 80% training, 20% testing)
split_index = int(len(inter_event_times) * 0.8)

# Split inter-event times into training and testing sets
inter_event_times_train = inter_event_times[:split_index].reset_index(drop=True)
inter_event_times_test = inter_event_times[split_index:].reset_index(drop=True)

# -----------------------------------
# Step 3: Fit Probability Distributions Using MLE
# -----------------------------------

# Fit the Exponential distribution to the training data
exp_params_train = stats.expon.fit(inter_event_times_train)

# Fit the Weibull distribution to the training data
weibull_params_train = stats.weibull_min.fit(inter_event_times_train)

# Fit the Gamma distribution to the training data
gamma_params_train = stats.gamma.fit(inter_event_times_train)

# Fit the Log-Normal distribution to the training data
lognorm_params_train = stats.lognorm.fit(inter_event_times_train, floc=0)

# -----------------------------------
# Step 4: Perform Goodness-of-Fit Tests
# -----------------------------------

# Create a dictionary of distributions and their fitted parameters
distributions = {
    'Exponential': ('expon', exp_params_train),
    'Weibull': ('weibull_min', weibull_params_train),
    'Gamma': ('gamma', gamma_params_train),
    'Log-Normal': ('lognorm', lognorm_params_train)
}

# Initialize a list to store test results
results_list = []

# Number of bins for Chi-Square test (rule of thumb)
num_bins = int(np.sqrt(len(inter_event_times_train)))

# Perform tests for each distribution
for name, (dist_name, params) in distributions.items():
    # K-S Test
    ks_statistic, ks_p_value = ks_test(inter_event_times_train, dist_name, params)
    
    # Anderson-Darling Test
    ad_statistic, _, _ = ad_test(inter_event_times_train, dist_name)
    
    # Chi-Square Goodness-of-Fit Test
    chi_statistic, chi_p_value = chi_square_test(inter_event_times_train, dist_name, params, num_bins)
    
    # Calculate Log-Likelihood
    pdf_vals = getattr(stats, dist_name).pdf(inter_event_times_train, *params)
    pdf_vals[pdf_vals == 0] = 1e-8
    log_likelihood = np.sum(np.log(pdf_vals))
    k = len(params)
    n = len(inter_event_times_train)
    aic = calculate_aic(log_likelihood, k)
    bic = calculate_bic(log_likelihood, k, n)
    
    # Prepare the new row as a dictionary
    new_row = {
        'Model': name,
        'K-S Statistic': ks_statistic,
        'K-S p-value': ks_p_value,
        'A-D Statistic': ad_statistic,
        'Chi-Square Statistic': chi_statistic,
        'Chi-Square p-value': chi_p_value,
        'AIC': aic,
        'BIC': bic
    }
    
    # Append the dictionary to the list
    results_list.append(new_row)

# After the loop, create the DataFrame
test_results = pd.DataFrame(results_list)

# Display Goodness-of-Fit test results
print("Goodness-of-Fit Test Results on Training Data:")
print(test_results[['Model', 'K-S Statistic', 'K-S p-value', 'A-D Statistic',
                    'Chi-Square Statistic', 'Chi-Square p-value', 'AIC', 'BIC']])

# -----------------------------------
# Step 5: Generate Predictions for Test Data
# -----------------------------------

# Number of test samples
n_test = len(inter_event_times_test)

# Generate predictions using the fitted distributions
# Exponential distribution predictions
predicted_exp = stats.expon.rvs(*exp_params_train, size=n_test)

# Weibull distribution predictions
predicted_weibull = stats.weibull_min.rvs(*weibull_params_train, size=n_test)

# Gamma distribution predictions
predicted_gamma = stats.gamma.rvs(*gamma_params_train, size=n_test)

# Log-Normal distribution predictions
predicted_lognorm = stats.lognorm.rvs(*lognorm_params_train, size=n_test)

# -----------------------------------
# Step 6: Evaluate Predictions Using Error Metrics
# -----------------------------------

# Define a function to compute evaluation metrics
def evaluate_predictions(actual, predicted):
    mae = mean_absolute_error(actual, predicted)                          # Mean Absolute Error
    rmse = np.sqrt(mean_squared_error(actual, predicted))                 # Root Mean Squared Error
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100           # Mean Absolute Percentage Error
    return mae, rmse, mape

# Evaluate predictions for each distribution
mae_exp, rmse_exp, mape_exp = evaluate_predictions(inter_event_times_test, predicted_exp)
mae_weibull, rmse_weibull, mape_weibull = evaluate_predictions(inter_event_times_test, predicted_weibull)
mae_gamma, rmse_gamma, mape_gamma = evaluate_predictions(inter_event_times_test, predicted_gamma)
mae_lognorm, rmse_lognorm, mape_lognorm = evaluate_predictions(inter_event_times_test, predicted_lognorm)

# Compile evaluation results into a DataFrame
evaluation_results = pd.DataFrame({
    'Model': ['Exponential', 'Weibull', 'Gamma', 'Log-Normal'],
    'MAE': [mae_exp, mae_weibull, mae_gamma, mae_lognorm],
    'RMSE': [rmse_exp, rmse_weibull, rmse_gamma, rmse_lognorm],
    'MAPE (%)': [mape_exp, mape_weibull, mape_gamma, mape_lognorm]
})

print("\nEvaluation Results on Test Data:")
print(evaluation_results)

# -----------------------------------
# Step 7: Visualize Actual vs. Predicted Inter-Event Times
# -----------------------------------

# Create a list of tuples containing model names and their predictions
model_predictions = [
    ('Exponential', predicted_exp),
    ('Weibull', predicted_weibull),
    ('Gamma', predicted_gamma),
    ('Log-Normal', predicted_lognorm)
]

# Loop over each model to create individual plots
for model_name, predictions in model_predictions:
    plt.figure(figsize=(12, 6))
    plt.plot(inter_event_times_test.values, label='Actual Inter-Event Times', marker='o')
    plt.plot(predictions, label=f'{model_name} Prediction', marker='x')
    plt.legend()
    plt.xlabel('Event Index in Test Set')
    plt.ylabel('Inter-Event Time (hours)')
    plt.title(f'Actual vs. Predicted Inter-Event Times - {model_name}')
    plt.show()

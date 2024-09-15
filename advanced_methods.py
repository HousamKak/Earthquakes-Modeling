# advanced_methods.py

"""
Earthquake Inter-Event Time Modeling - Advanced Methods

This script implements the following advanced modeling approaches:
1. Hawkes Process
2. Simplified Epidemic-Type Aftershock Sequence (ETAS) Model
3. Time-Varying Poisson Model

Uses statistical tests from statistical_tests.py
"""

# -----------------------------------
# Import Necessary Libraries
# -----------------------------------

import pandas as pd                # For data manipulation
import numpy as np                 # For numerical computations
from scipy import stats            # For statistical distributions and tests
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
df = pd.read_csv('earthquake_data.csv')

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

# Determine the split index (e.g., 80% training, 20% testing)
split_index = int(len(inter_event_times) * 0.8)

# Split inter-event times into training and testing sets
inter_event_times_train = inter_event_times[:split_index].reset_index(drop=True)
inter_event_times_test = inter_event_times[split_index:].reset_index(drop=True)

# -----------------------------------
# Define Evaluation Function
# -----------------------------------

# Define a function to compute evaluation metrics
def evaluate_predictions(actual, predicted):
    mae = mean_absolute_error(actual, predicted)                          # Mean Absolute Error
    rmse = np.sqrt(mean_squared_error(actual, predicted))                 # Root Mean Squared Error
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100           # Mean Absolute Percentage Error
    return mae, rmse, mape

# Initialize evaluation results DataFrame
evaluation_results = pd.DataFrame(columns=['Model', 'MAE', 'RMSE', 'MAPE (%)'])

# -----------------------------------
# Advanced Modeling Approaches
# -----------------------------------

# ===================================
# Hawkes Process Implementation
# ===================================

# -----------------------------------
# Step A1: Implementing a Basic Hawkes Process
# -----------------------------------

# Define the simulation parameters
mu = 0.1       # Background intensity (events per hour)
alpha = 0.5    # Excitation parameter
beta = 1.0     # Decay rate of the exponential kernel

# Initialize lists to store event times
event_times_hawkes = []

# Start time
t = 0

# Seed the random number generator for reproducibility
np.random.seed(42)

# Simulate the Hawkes process
while len(event_times_hawkes) < len(inter_event_times_test):
    # Compute the conditional intensity function
    if len(event_times_hawkes) == 0:
        lambda_t = mu
    else:
        past_events = np.array(event_times_hawkes)
        lambda_t = mu + np.sum(alpha * beta * np.exp(-beta * (t - past_events)))
    
    # Generate the next event time using an exponential distribution
    u = np.random.uniform(0, 1)
    w = -np.log(u) / lambda_t
    t += w
    
    # Append the event time
    event_times_hawkes.append(t)

# Calculate inter-event times from the simulated event times
inter_event_times_hawkes = np.diff([0] + event_times_hawkes)  # Include start time 0

# Ensure the number of simulated inter-event times matches the test set
inter_event_times_hawkes = inter_event_times_hawkes[:len(inter_event_times_test)]

# -----------------------------------
# Step A2: Evaluate Hawkes Process Predictions
# -----------------------------------

# Evaluate predictions using the same metrics
mae_hawkes, rmse_hawkes, mape_hawkes = evaluate_predictions(
    inter_event_times_test,
    inter_event_times_hawkes
)

# Add Hawkes process results to the evaluation DataFrame
new_row = {
    'Model': 'Hawkes Process',
    'MAE': mae_hawkes,
    'RMSE': rmse_hawkes,
    'MAPE (%)': mape_hawkes
}
evaluation_results = pd.concat([evaluation_results, pd.DataFrame([new_row])], ignore_index=True)

print("\nEvaluation Results Including Hawkes Process:")
print(evaluation_results)

# -----------------------------------
# Step A3: Visualize Hawkes Process Predictions
# -----------------------------------

plt.figure(figsize=(12, 6))
plt.plot(inter_event_times_test.values, label='Actual Inter-Event Times', marker='o')
plt.plot(inter_event_times_hawkes, label='Hawkes Process Prediction', marker='x')
plt.legend()
plt.xlabel('Event Index in Test Set')
plt.ylabel('Inter-Event Time (hours)')
plt.title('Actual vs. Hawkes Process Predicted Inter-Event Times')
plt.show()

# ===================================
# Simplified ETAS Model Implementation
# ===================================

# -----------------------------------
# Step B1: Check if 'magnitude' Column Exists
# -----------------------------------

# Ensure that the 'magnitude' column exists
if 'magnitude' in df.columns:
    magnitudes = df['magnitude'].dropna().reset_index(drop=True)
    etas_model_available = True
else:
    # If magnitudes are not available, we cannot proceed with ETAS
    print("\nMagnitudes are not available in the dataset. ETAS model cannot be applied.")
    etas_model_available = False

if etas_model_available:
    # -----------------------------------
    # Step B2: Implementing a Simplified ETAS Model
    # -----------------------------------

    # Define the simulation parameters
    mu = 0.1        # Background intensity
    beta = 1.0      # Decay rate
    c = 0.001       # Time offset
    p = 1.1         # Decay exponent
    K = 0.5         # Productivity constant
    alpha_m = 1.0   # Magnitude scaling factor
    M0 = magnitudes.min()  # Minimum magnitude

    # Initialize lists to store event times and magnitudes
    event_times_etas = []
    event_magnitudes_etas = []

    # Start time
    t = 0

    # Seed the random number generator for reproducibility
    np.random.seed(42)

    # Simulate the ETAS process
    while len(event_times_etas) < len(inter_event_times_test):
        # Compute the total intensity
        lambda_t = mu
        for ti, mi in zip(event_times_etas, event_magnitudes_etas):
            if t > ti:
                lambda_t += K * np.exp(alpha_m * (mi - M0)) * (1 / ((t - ti + c) ** p))

        # Generate the next event time
        u = np.random.uniform(0, 1)
        w = -np.log(u) / lambda_t
        t += w

        # Append the event time
        event_times_etas.append(t)

        # Generate a magnitude for the event (using Gutenberg-Richter law)
        m = np.random.exponential(scale=1.0) + M0  # Scale parameter can be adjusted
        event_magnitudes_etas.append(m)

    # Calculate inter-event times from the simulated event times
    inter_event_times_etas = np.diff([0] + event_times_etas)  # Include start time 0

    # Ensure the number of simulated inter-event times matches the test set
    inter_event_times_etas = inter_event_times_etas[:len(inter_event_times_test)]

    # -----------------------------------
    # Step B3: Evaluate ETAS Model Predictions
    # -----------------------------------

    # Evaluate predictions using the same metrics
    mae_etas, rmse_etas, mape_etas = evaluate_predictions(
        inter_event_times_test,
        inter_event_times_etas
    )

    # Add ETAS model results to the evaluation DataFrame
    new_row = {
        'Model': 'Simplified ETAS Model',
        'MAE': mae_etas,
        'RMSE': rmse_etas,
        'MAPE (%)': mape_etas
    }
    evaluation_results = pd.concat([evaluation_results, pd.DataFrame([new_row])], ignore_index=True)

    print("\nEvaluation Results Including Simplified ETAS Model:")
    print(evaluation_results)

    # -----------------------------------
    # Step B4: Visualize ETAS Model Predictions
    # -----------------------------------

    plt.figure(figsize=(12, 6))
    plt.plot(inter_event_times_test.values, label='Actual Inter-Event Times', marker='o')
    plt.plot(inter_event_times_etas, label='Simplified ETAS Model Prediction', marker='x')
    plt.legend()
    plt.xlabel('Event Index in Test Set')
    plt.ylabel('Inter-Event Time (hours)')
    plt.title('Actual vs. Simplified ETAS Model Predicted Inter-Event Times')
    plt.show()

# ===================================
# Time-Varying Poisson Model Implementation
# ===================================

# -----------------------------------
# Step C1: Estimate Time-Varying Event Rate
# -----------------------------------

# Define the window size (number of events) for moving average
window_size = 50  # Adjust based on data size

# Initialize lists to store estimated lambda values and corresponding times
lambda_t = []
time_t = []

# Calculate the time-varying lambda using a moving window
for i in range(len(inter_event_times_train) - window_size + 1):
    window_events = inter_event_times_train[i:i+window_size]
    total_time_window = window_events.sum()
    lambda_window = window_size / total_time_window  # Events per hour
    lambda_t.append(lambda_window)
    # Use the midpoint time of the window
    time_t.append(df['time'].iloc[i + window_size // 2])

# -----------------------------------
# Step C2: Plot Time-Varying Event Rate
# -----------------------------------

plt.figure(figsize=(12, 6))
plt.plot(time_t, lambda_t, marker='o')
plt.xlabel('Time')
plt.ylabel('Estimated Lambda (events per hour)')
plt.title('Time-Varying Estimated Lambda')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# -----------------------------------
# Step C3: Predict Expected Number of Events in Test Period
# -----------------------------------

# Use the average lambda from the last window as the rate for the test period
lambda_test = lambda_t[-1]

# Total time in the test period
total_time_test = inter_event_times_test.sum()

# Expected number of events in the test period
expected_events_tv = lambda_test * total_time_test

# Actual number of events in test set
actual_events_test = len(inter_event_times_test)

print(f"\nTime-Varying Model Expected Number of Events: {expected_events_tv:.2f}")
print(f"Actual Number of Events in Test Period: {actual_events_test}")

# Calculate error metrics for counts
count_error = abs(expected_events_tv - actual_events_test)
count_percentage_error = (count_error / actual_events_test) * 100

# Add Time-Varying model results to the evaluation DataFrame
new_row = {
    'Model': 'Time-Varying Poisson',
    'MAE': count_error,
    'RMSE': np.sqrt(count_error**2),
    'MAPE (%)': count_percentage_error
}
evaluation_results = pd.concat([evaluation_results, pd.DataFrame([new_row])], ignore_index=True)

print("\nEvaluation Results Including Time-Varying Poisson Model:")
print(evaluation_results)

# -----------------------------------
# Step C4: Visualize Time-Varying Poisson Model Predictions
# -----------------------------------

# Create a time series of cumulative counts for the actual test data
cumulative_counts_actual = np.arange(1, len(inter_event_times_test) + 1)
cumulative_time_actual = np.cumsum(inter_event_times_test)

# Debugging: Check if inter_event_times_test and cumulative_time_actual are populated correctly
print(f"Length of inter_event_times_test: {len(inter_event_times_test)}")
print(f"inter_event_times_test: {inter_event_times_test}")
print(f"cumulative_time_actual: {cumulative_time_actual}")

# Ensure cumulative_time_actual is not empty
if len(cumulative_time_actual) == 0:
    raise ValueError("cumulative_time_actual is empty. Ensure that inter_event_times_test has valid values.")

# Create a time series of cumulative counts for the expected counts based on the model
expected_event_rate = lambda_test  # Events per hour

# Ensure cumulative_time_actual has values before calculating max_time
if len(cumulative_time_actual) > 0:
    max_time = cumulative_time_actual.iloc[-1]  # Use .iloc[-1] to get the last element safely
else:
    max_time = 0

# Generate expected cumulative counts over the test period
time_steps = np.linspace(0, max_time, len(inter_event_times_test))
cumulative_counts_expected = expected_event_rate * time_steps

# Plot the cumulative counts
plt.figure(figsize=(12, 6))
plt.plot(cumulative_time_actual, cumulative_counts_actual, label='Actual Cumulative Counts', marker='o')
plt.plot(time_steps, cumulative_counts_expected, label='Predicted Cumulative Counts (Time-Varying Poisson)', marker='x')
plt.legend()
plt.xlabel('Time (hours)')
plt.ylabel('Cumulative Event Count')
plt.title('Actual vs. Predicted Cumulative Counts - Time-Varying Poisson Model')
plt.show()


# -----------------------------------
# Final Evaluation and Comparison
# -----------------------------------

# Display the final evaluation results
print("\nFinal Evaluation Results:")
print(evaluation_results)

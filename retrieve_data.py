# retrieve_data.py

"""
Script to Retrieve Earthquake Data from USGS Earthquake API

This script fetches earthquake data from the USGS Earthquake API based on specified parameters
and saves the data to a CSV file named 'earthquake_data.csv'.

Parameters:
- Start time
- End time
- Minimum magnitude
- Order by time ascending

The resulting CSV file can be used with the earthquake inter-event time modeling scripts.
"""

import requests
import pandas as pd

# -----------------------------------
# Step 1: Define the API Endpoint and Parameters
# -----------------------------------

# USGS Earthquake API endpoint
url = 'https://earthquake.usgs.gov/fdsnws/event/1/query'

# Define the parameters for the API request
params = {
    'format': 'csv',                # Request CSV format
    'starttime': '2023-08-01',      # Start date (YYYY-MM-DD)
    'endtime': '2023-09-01',        # End date (YYYY-MM-DD)
    'minmagnitude': 2.5,            # Minimum magnitude to include
    'orderby': 'time-asc'           # Order events by time ascending
}

# -----------------------------------
# Step 2: Make the API Request
# -----------------------------------

# Send the GET request to the API
response = requests.get(url, params=params)

# Check if the request was successful
if response.status_code == 200:
    print('Data retrieved successfully!')
else:
    print(f'Error fetching data. HTTP Status code: {response.status_code}')
    exit()

# -----------------------------------
# Step 3: Save the Data to a CSV File
# -----------------------------------

# Save the CSV data to a file with UTF-8 encoding
with open('earthquake_data.csv', 'w', encoding='utf-8') as f:
    f.write(response.text)

print('Data saved to earthquake_data.csv')

# -----------------------------------
# Step 4: Load and Inspect the Data
# -----------------------------------

# Read the CSV data into a pandas DataFrame
df = pd.read_csv('earthquake_data.csv')

# Display the first few rows
print('\nFirst few rows of the data:')
print(df.head())

# Display the columns
print('\nColumns in the data:')
print(df.columns)

# retrieve_data.py

"""
Script to Retrieve Earthquake Data from USGS Earthquake API by Month

This script fetches earthquake data from the USGS Earthquake API in monthly chunks 
and appends the results into a CSV file named 'earthquake_data.csv'.

Parameters:
- Start year and month
- Number of months to fetch
- Minimum magnitude
"""

import requests
import pandas as pd
from datetime import datetime, timedelta

# -----------------------------------
# Step 1: Define a function to fetch data for a single month
# -----------------------------------

def fetch_data(start_time, end_time, min_magnitude=2.5, filename='earthquake_data.csv'):
    """
    Fetches earthquake data for the specified time period and appends it to the given CSV file.
    
    Args:
        start_time (str): The start date in YYYY-MM-DD format.
        end_time (str): The end date in YYYY-MM-DD format.
        min_magnitude (float): The minimum magnitude of earthquakes to fetch.
        filename (str): The name of the file to save data to.
    """
    # USGS Earthquake API endpoint
    url = 'https://earthquake.usgs.gov/fdsnws/event/1/query'
    
    # Define the parameters for the API request
    params = {
        'format': 'csv',
        'starttime': start_time,
        'endtime': end_time,
        'minmagnitude': min_magnitude,
        'orderby': 'time-asc'
    }
    
    # Send the GET request to the API
    response = requests.get(url, params=params)
    
    # Check if the request was successful
    if response.status_code == 200:
        print(f'Data retrieved for {start_time} to {end_time} successfully!')
        
        # Append the data to the CSV file
        with open(filename, 'a', encoding='utf-8') as f:
            f.write(response.text)
    else:
        print(f'Error fetching data for {start_time} to {end_time}. HTTP Status code: {response.status_code}')


# -----------------------------------
# Step 2: Fetch earthquake data in chunks by month
# -----------------------------------

def fetch_data_by_month(start_year, start_month, num_months, min_magnitude=2.5, output_csv='earthquake_data.csv'):
    """
    Fetches earthquake data by iterating through each month and appending the data to a CSV file.
    
    Args:
        start_year (int): The starting year for data retrieval.
        start_month (int): The starting month for data retrieval.
        num_months (int): The number of months of data to retrieve.
        min_magnitude (float): The minimum magnitude of earthquakes to fetch.
        output_csv (str): The name of the output CSV file.
    """
    # Start by creating an empty file with headers for the first request
    with open(output_csv, 'w', encoding='utf-8') as f:
        f.write('')  # Create an empty file to append to later
    
    # Calculate the start date and loop over the number of months
    current_date = datetime(year=start_year, month=start_month, day=1)
    for _ in range(num_months):
        # Set the start and end time for the current month
        start_time = current_date.strftime('%Y-%m-%d')
        next_month = current_date + timedelta(days=32)  # Move to the next month
        next_month = next_month.replace(day=1)  # Set to the first day of the next month
        end_time = (next_month - timedelta(days=1)).strftime('%Y-%m-%d')  # Set to the last day of the current month

        # If the current month is the current month, adjust the end time to today
        if next_month > datetime.now():
            end_time = datetime.now().strftime('%Y-%m-%d')

        # Fetch the data for the current month
        fetch_data(start_time, end_time, min_magnitude=min_magnitude, filename=output_csv)

        # Move to the next month
        current_date = next_month

    print(f'\nAll data saved to {output_csv}')


# -----------------------------------
# Step 3: Main Execution Logic
# -----------------------------------

if __name__ == "__main__":
    # # Specify the starting year and month for data retrieval
    # start_year = int(input("Enter the starting year (e.g., 2023): "))
    # start_month = int(input("Enter the starting month (1-12): "))
    # num_months = int(input("Enter the number of months to retrieve: "))
    
    # # Specify the minimum magnitude
    # min_magnitude = float(input("Enter the minimum magnitude to filter by (e.g., 2.5): "))
    
    
        # Specify the starting year and month for data retrieval
    start_year = 2000
    start_month = 1
    num_months = 120
    
    # Specify the minimum magnitude
    min_magnitude = 0

    # Call the function to retrieve data by month and save to a CSV file
    fetch_data_by_month(start_year, start_month, num_months, min_magnitude=min_magnitude, output_csv='src/data/earthquake_data.csv')

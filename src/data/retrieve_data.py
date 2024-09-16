import requests
import pandas as pd
from datetime import datetime, timedelta

def fetch_data(start_time, end_time, min_magnitude=2.5, filename='earthquake_data.csv', write_header=False):
    """
    Fetches earthquake data for the specified time period and appends it to the given CSV file.
    
    Args:
        start_time (str): The start date in YYYY-MM-DD format.
        end_time (str): The end date in YYYY-MM-DD format.
        min_magnitude (float): The minimum magnitude of earthquakes to fetch.
        filename (str): The name of the file to save data to.
        write_header (bool): Whether to write headers or not.
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
        
        # Convert the response content to a pandas DataFrame
        from io import StringIO
        csv_data = StringIO(response.text)
        df = pd.read_csv(csv_data)
        
        # Append the data to the CSV file
        df.to_csv(filename, mode='a', header=write_header, index=False)
    else:
        print(f'Error fetching data for {start_time} to {end_time}. HTTP Status code: {response.status_code}')

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
    current_date = datetime(year=start_year, month=start_month, day=1)
    first_fetch = True  # Flag to control writing the header only once

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
        fetch_data(start_time, end_time, min_magnitude=min_magnitude, filename=output_csv, write_header=first_fetch)

        # After the first fetch, set write_header to False
        first_fetch = False

        # Move to the next month
        current_date = next_month

    print(f'\nAll data saved to {output_csv}')

if __name__ == "__main__":
    
 # # Specify the starting year and month for data retrieval
    # start_year = int(input("Enter the starting year (e.g., 2023): "))
    # start_month = int(input("Enter the starting month (1-12): "))
    # num_months = int(input("Enter the number of months to retrieve: "))

    # # Specify the minimum magnitude
    # min_magnitude = float(input("Enter the minimum magnitude to filter by (e.g., 2.5): "))
    # Specify the starting year and month for data 

    start_year = 2000
    start_month = 1
    num_months = 2
    
    # Specify the minimum magnitude
    min_magnitude = 0

    # Call the function to retrieve data by month and save to a CSV file
    fetch_data_by_month(start_year, start_month, num_months, min_magnitude=min_magnitude, output_csv='src/data/earthquake_data.csv')

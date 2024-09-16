import pandas as pd
import numpy as np

class FeatureEngineering:
    def __init__(self, window_size=10):
        self.window_size = window_size

    @staticmethod
    def compute_gutenberg_richter_slope(magnitudes):
        """
        Computes the slope (b-value) of the Gutenberg-Richter law.
        """
        # Convert all values to numeric, forcing errors to NaN
        magnitudes = pd.to_numeric(magnitudes, errors='coerce')

        # Filter out NaN values (non-numeric or missing data)
        magnitudes = magnitudes[~np.isnan(magnitudes)]
        
        if len(magnitudes) > 1:
            sorted_mags = np.sort(magnitudes)
            N = np.array([len(sorted_mags[sorted_mags >= mag]) for mag in sorted_mags])
            log_N = np.log10(N)
            valid = ~np.isinf(log_N)
            if np.sum(valid) > 1:
                slope, intercept = np.polyfit(magnitudes[valid], log_N[valid], 1)
                return -slope  # b-value
        return 0.0

    def extract_features(self, df):
        """
        Extracts seismicity indicators as per the paper.
        """
        # Convert the 'time' column to datetime, using errors='coerce' to handle invalid formats
        df.loc[:, 'time'] = pd.to_datetime(df['time'], errors='coerce')

        # Drop rows where 'time' is NaT (invalid times)
        df = df.dropna(subset=['time'])

        # Convert 'mag' column to numeric, forcing errors to NaN using .loc to avoid SettingWithCopyWarning
        df = df.copy()  # Explicitly create a copy of the DataFrame
        df.loc[:, 'mag'] = pd.to_numeric(df['mag'], errors='coerce')

        # Drop rows with NaN magnitudes
        df = df.dropna(subset=['mag'])

        df = df.sort_values('time').reset_index(drop=True)

        features = {
            'elapsed_time': [],
            'slope_b': [],
            'mean_square_dev': [],
            'avg_magnitude': [],
            'magnitude_deficit': [],
            'seismic_energy_rate': [],
            'mean_time_between_events': [],
            'coefficient_variation': []
        }

        targets = []

        for i in range(self.window_size, len(df)):
            past_events = df.iloc[i - self.window_size:i]

            delta_t = (past_events['time'].iloc[-1] - past_events['time'].iloc[0]).total_seconds() / 86400
            features['elapsed_time'].append(delta_t)

            b_value = self.compute_gutenberg_richter_slope(past_events['mag'].values)
            features['slope_b'].append(b_value)

            if b_value > 0:
                sorted_mags = np.sort(past_events['mag'].values)
                N = np.array([len(sorted_mags[sorted_mags >= mag]) for mag in sorted_mags])
                log_N = np.log10(N)
                slope, intercept = np.polyfit(past_events['mag'], log_N, 1)
                predicted_log_N = slope * past_events['mag'] + intercept
                mse = np.mean((log_N - predicted_log_N) ** 2)
            else:
                mse = 0.0
            features['mean_square_dev'].append(mse)

            Mmean = past_events['mag'].mean()
            features['avg_magnitude'].append(Mmean)

            max_mag = past_events['mag'].max()
            expected_mag = b_value * np.log10(self.window_size) if b_value > 0 else 0.0
            delta_M = max_mag - expected_mag
            features['magnitude_deficit'].append(delta_M)

            energy = np.sum(10 ** (1.5 * past_events['mag'].values))
            sqrt_energy = np.sqrt(energy)
            features['seismic_energy_rate'].append(sqrt_energy)

            inter_event_times = past_events['time'].diff().dt.total_seconds().dropna() / 86400
            if len(inter_event_times) > 0:
                mu = inter_event_times.mean()
            else:
                mu = 0.0
            features['mean_time_between_events'].append(mu)

            if mu > 0 and len(inter_event_times) > 1:
                c = inter_event_times.std() / mu
            else:
                c = 0.0
            features['coefficient_variation'].append(c)

            current_mag = df['mag'].iloc[i]
            if current_mag < 4.5:
                mag_class = 1
            elif 4.5 <= current_mag < 5.0:
                mag_class = 2
            elif 5.0 <= current_mag < 5.5:
                mag_class = 3
            elif 5.5 <= current_mag < 6.0:
                mag_class = 4
            elif 6.0 <= current_mag < 6.5:
                mag_class = 5
            elif 6.5 <= current_mag < 7.0:
                mag_class = 6
            else:
                mag_class = 7
            targets.append(mag_class)

        features_df = pd.DataFrame(features)
        features_df['target'] = targets

        return features_df

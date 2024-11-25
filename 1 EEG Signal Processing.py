import os
import re
import mne
import pandas as pd
import numpy as np
from mne.time_frequency import tfr_array_morlet
from scipy.stats import kurtosis
from mne.preprocessing import ICA
from scipy.signal import coherence
import warnings
import json

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Define directories
output_directory = "E:/ESPERIMENTI DISMI/IEEE/EEG_Enobio20_Signals_Filtered_Coherence"
output_directory_csv = os.path.join(output_directory, "CSV")
output_directory_fif = os.path.join(output_directory, "FIF")
input_directory = "E:/ESPERIMENTI DISMI/Datasets DISMI Correct"

# Create output directories if they don't exist
os.makedirs(output_directory_csv, exist_ok=True)
os.makedirs(output_directory_fif, exist_ok=True)

class Patient:
    def __init__(self, filepath, patient_id, task_number):
        self.id = patient_id
        self.task_number = task_number
        self.filepath = filepath
        self.df = None
        self.raw = None
        self.columns = []

    def load_data(self):
        try:
            if self.task_number != 0:
                self.df = pd.read_csv(self.filepath, sep=',', usecols=range(21), low_memory=False)
            else:
                self.df = pd.read_csv(self.filepath, sep=',', low_memory=False)
                valid_source_names = ["60 Sec Video", "60 Sec Video-1", "60 Sec Video-2"]
                self.df = self.df[self.df['SourceStimuliName'].isin(valid_source_names)]
                self.df.drop(['SourceStimuliName'], axis=1, inplace=True)
        except Exception as e:
            raise Exception(f"Error loading data: {e}")

        self.columns = self.df.columns.tolist()[1:]

    def convert_timestamp(self):
        try:
            self.df['Timestamp'] = pd.to_datetime(self.df['Timestamp'], format="%Y-%m-%d %H:%M:%S.%f")
        except ValueError:
            self.df['Timestamp'] = pd.to_datetime(self.df['Timestamp'], format="%M:%S.%f")
        self.df['Timestamp'] = self.df['Timestamp'].astype(np.int64) / 10**6

    def filter_data(self):
        df = self.df[self.columns].to_numpy()
        info = mne.create_info(ch_names=self.columns, sfreq=500, ch_types='misc')
        self.raw = mne.io.RawArray(df.T, info, verbose=False)

        type_dict = {ch: 'eeg' if ch not in ['EOG1', 'EOG2'] else 'eog' for ch in self.columns}
        del type_dict['EXT']
        self.raw.set_channel_types(type_dict, verbose=False)

        total_time = self.raw.times[-1]  # Ultimo timestamp
        if total_time > 5:
            self.raw.crop(tmin=5, tmax=total_time - 5)
        else:
            print(f"Warning: Insufficient data length for patient {self.id} in task {self.task_number}. Creating an empty DataFrame.")
            self.raw = None  # Imposta a None o un oggetto vuoto
            return pd.DataFrame()  # Restituisci un DataFrame vuoto

        self.raw.resample(250)
        self.eog = self.raw.copy().pick(['EOG1', 'EOG2', 'EXT'])
        self.raw.drop_channels(['EOG1', 'EOG2', 'EXT'])
        self.raw.filter(1, 30, picks=self.raw.ch_names, verbose=False)
        mne.baseline.rescale(self.raw._data, self.raw.times, baseline=(None, 0), verbose=False)

    def remove_bad_channels(self):
        kurt = kurtosis(self.raw.get_data(), axis=1)
        bad_channels = np.where(kurt > 5)[0]
        self.raw.info['bads'] = [self.raw.ch_names[i] for i in bad_channels]
        self.raw.drop_channels(self.raw.info['bads'])

        n_channels = len(self.raw.ch_names)
        print(f"Number of channels: {n_channels}")
        ica = ICA(n_components=n_channels, random_state=97, verbose=False)
        ica.fit(self.raw, picks=self.raw.ch_names, verbose=False)
        ica.apply(self.raw, verbose=False)
        self.raw.apply_function(lambda x: (x - np.mean(x)) / np.std(x))

    def get_power_spectrum(self, window_length):
        channels = self.raw.ch_names
        power_dict = {f"{key}_{band}": [] for key in channels for band in ['theta', 'alpha', 'beta']}
        coherence_dict = {f"{key1}_{key2}_coherence": [] for key1 in channels for key2 in channels if key1 != key2}

        bands = {"theta": (4, 8), "alpha": (8, 13), "beta": (13, 30)}
        n_samples_window = int(window_length * self.raw.info['sfreq'])
        n_windows = len(self.raw.times) // n_samples_window

        for i in range(n_windows):
            start = i * n_samples_window
            end = (i + 1) * n_samples_window
            data_window = self.raw.get_data(start=start, stop=end)

            # Calculate power spectrum for each band
            for band_name, (fmin, fmax) in bands.items():
                power = tfr_array_morlet(data_window[np.newaxis, :, :], sfreq=self.raw.info['sfreq'],
                                          freqs=np.arange(fmin, fmax + 1), output='power', zero_mean=False, verbose=False)
                avg_power = np.mean(power, axis=2).squeeze()
                # Calcola la media dell'array power per ogni canale
                avg_power = np.mean(avg_power, axis=1)
                for j, channel in enumerate(channels):
                    power_dict[f'{channel}_{band_name}'].append(avg_power[j])

            # Calculate coherence between pairs of channels
            for j in range(len(channels)):
                for k in range(j + 1, len(channels)):
                    f, Cxy = coherence(data_window[j], data_window[k], fs=self.raw.info['sfreq'])
                    avg_coherence = np.mean(Cxy) if len(Cxy) > 0 else np.nan  # Handle empty results
                    coherence_dict[f'{channels[j]}_{channels[k]}_coherence'].append(avg_coherence)

        # Ensure all lists are the same length by filling with NaN
        max_length = n_windows
        for key in coherence_dict.keys():
            while len(coherence_dict[key]) < max_length:
                coherence_dict[key].append(np.nan)

        task = f'Task {self.task_number}' if self.task_number != 0 else "Baseline"
        new_df_power = pd.DataFrame(power_dict)
        new_df_coherence = pd.DataFrame(coherence_dict)

        new_df_coherence.dropna(axis=1, how='all', inplace=True)

        # Concatena i due dataset
        new_df_power = pd.concat([new_df_power, new_df_coherence], axis=1)

        # Handle timestamps
        time_stamp = list(range(window_length, window_length * (n_windows + 1), window_length))
        new_df_power.insert(0, "Timestamp", time_stamp)
        new_df_coherence.insert(0, "Timestamp", time_stamp)

        # Save power and coherence results to separate CSV files
        output_power_path = os.path.join(output_directory_csv, f'EEG_{task}_P_{self.id:02}.csv')
        new_df_power.to_csv(output_power_path, index=False)

        output_fif_path = os.path.join(output_directory_fif, f'EEG_{task}_P_{self.id:02}_raw.fif')
        self.raw.save(output_fif_path, overwrite=True, verbose=False)

        return power_dict, coherence_dict

if __name__ == "__main__":
    bad_files = []
    for file_name in os.listdir(os.path.abspath(input_directory)):
        if file_name.lower().endswith('.csv') and "EEG" in file_name and "Task 5_P_05" not in file_name:
            file_path = os.path.join(input_directory, file_name)
            find_task = re.search(r"Task (\d+)", file_name)
            task_number = int(find_task.group(1)) if find_task else 0
            find_patient = re.search(r"P_(\d+)", file_name)
            patient_number = int(find_patient.group(1)) if find_patient else None

            patient = Patient(file_path, patient_number, task_number)
            print(f"Analyzing {file_name}...")
            patient.load_data()
            patient.convert_timestamp()
            patient.filter_data()

            try:
                patient.remove_bad_channels()
            except ValueError:
                print(f"All channels are 'bad' for {file_name}")
                bad_files.append(file_path)
                continue

            patient.get_power_spectrum(10)
            print("...completed successfully!\n")
        else:
            print(f"Error: {file_name} is not a valid CSV.")

    with open(os.path.join(output_directory, 'badFiles.json'), "w") as json_file:
        json.dump(bad_files, json_file)

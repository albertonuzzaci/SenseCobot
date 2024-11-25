import os
import numpy as np
import pywt
from scipy.signal import butter, filtfilt  
from math import isnan
from setup import setup, getConfigData
from Analysis.main_class import Participant
import pandas as pd
from scipy.signal import find_peaks
import neurokit2 as nk
from scipy.fft import fft, fftfreq

class Participant_GSR(Participant):
    
    def __init__(self, filepath):
        super().__init__(filepath)
        config_data = getConfigData()
        setup(config_data)
        self.output_directory = f"{config_data['PFOLDER']['GSR_PROCESSED']}"
        self.output_directory_windows = f"{config_data['PFOLDER']['GSR_PROCESSED_WINDOWS']}"
        self.output_directory_freq = f"{config_data['PFOLDER']['GSR_PROCESSED_FREQUENCY']}"
    
    def loadData(self):
        '''
        - Try to load the data from the file_name.
            If the task is the "Baseline" then keep only the rows corresponding to the blue video, 
            otherwise keep just the first 4 columns
        - Convert "Timestamp" column to milliseconds. 
        '''
        try:
            if self.tasknumber != 0: 
                self.df = pd.read_csv(self.filepath, 
                                    sep=',', 
                                    dtype={"GSR Resistance CAL": float, "GSR Conductance CAL": float},
                                    usecols=range(3)
                                    )
            else:
                self.df = pd.read_csv(self.filepath, 
                                    sep=',', 
                                    dtype={"GSR Resistance CAL": float, "GSR Conductance CAL": float, "SourceStimuliName":str}
                )
                valid_source_names = ["60 Sec Video", "60 Sec Video-1", "60 Sec Video-2"]
                self.df = self.df[self.df['SourceStimuliName'].isin(valid_source_names)]
                self.df = self.df.drop(['SourceStimuliName'], axis=1)
        except Exception as e:
            raise Exception(f"While loading the data this Expection occurred:\n{e} - {type(e)}")
        
        try:
            self.df['Timestamp'] = pd.to_datetime(self.df['Timestamp'], format="%Y-%m-%d %H:%M:%S.%f")
        except ValueError:
            self.df['Timestamp'] = pd.to_datetime(self.df['Timestamp'], format="%M:%S.%f")
        
        start_time = self.df['Timestamp'].iloc[0]
        self.df['Timestamp'] = (self.df['Timestamp'] - start_time).dt.total_seconds() * 1000
        
        self.df = self.df.drop(columns=['GSR Resistance CAL']) 
    
    
    def filterWaveletTrasform(self):
        wavelet = pywt.Wavelet('db5')
        coeffs = pywt.wavedec(self.df['GSR Conductance CAL'], wavelet)
        level = 4
        coeffs_filtered = coeffs[:level] + [None] * (len(coeffs) - level)
        self.df['GSR Conductance CAL'] = pywt.waverec(coeffs_filtered, wavelet)[:len(self.df)]
        self.df.dropna(inplace=True)
        
        
    def getThresholds(self, multiplier=2):
        '''
        Compute threshold for artifacts removing
        '''
        mean = self.df['GSR Conductance CAL'].mean() 
        std = self.df['GSR Conductance CAL'].std() 
        
        threshold_upper = mean + multiplier * std
        threshold_lower = mean - multiplier * std
        
        return threshold_lower, threshold_upper

    def removeArtifacts(self, threshold_lower, threshold_upper):
        '''
        Artifacts removal based on thresholds
        '''
        self.df = self.df[(self.df['GSR Conductance CAL'] < threshold_upper) & (self.df['GSR Conductance CAL'] > threshold_lower)]
        self.df = self.df.reset_index()
        self.df = self.df.drop(columns=['index'])
        
    def filterBandPass(self, fs):
        '''
        Apply a band pass filter to the GSR signal
        '''   
        f_low = 5           # Frequency of the low pass filter
        nyq = 0.5 * fs      # Nyquist frequency
        low = f_low / nyq   # Normalized frequency
        
        b, a = butter(4, low, btype='low')
        new_GSR_conductance = pd.DataFrame(filtfilt(b, a, self.df['GSR Conductance CAL']))
        self.df['GSR Conductance CAL'] = new_GSR_conductance[0]   
    
    def getTonicPhasic(self):
        '''
        Compute tonic and phasic components of the GSR signal
        '''
        data = nk.eda_phasic(nk.standardize(self.df['GSR Conductance CAL']), sampling_rate=512)
        self.df['GSR_phasic'] = data["EDA_Phasic"]
        self.df['GSR_tonic'] = data["EDA_Tonic"]
     
    def power_spectrum_analysis(self, fs):
        # Convert Pandas Series to NumPy array for FFT
        gsr_phasic_data = self.df['GSR_phasic'].to_numpy()

        # Compute FFT
        N = len(gsr_phasic_data)  # N è la lunghezza dell'intero segnale
        fft_values = fft(gsr_phasic_data)
        fft_freqs = fftfreq(N, 1/fs)

        # Focus on positive frequencies
        positive_freqs = fft_freqs[:N//2]  # Frequenze positive
        positive_fft_values = np.abs(fft_values[:N//2])

        # Compute Power Spectral Density (PSD)
        psd = (positive_fft_values**2) / N  # Potenza normalizzata per la lunghezza del segnale

        # Calcolare la potenza nelle bande di frequenza
        low_band = (0, 0.25) # 0-0.25 Hz
        mid_band = (0.045, 0.25) # 0.045-0.25 Hz
        high_band = (0, 0.4) # 0-0.4 Hz

        # Calcola la potenza per ogni banda
        low_band_power = np.sum(psd[(positive_freqs >= low_band[0]) & (positive_freqs <= low_band[1])])
        mid_band_power = np.sum(psd[(positive_freqs >= mid_band[0]) & (positive_freqs <= mid_band[1])])
        high_band_power = np.sum(psd[(positive_freqs >= high_band[0]) & (positive_freqs <= high_band[1])])

        # Salva i risultati dell'analisi dello spettro di potenza
        power_spectrum_data = {
            "Frequency (Hz)": positive_freqs,
            "Power (PSD)": psd,
            "Low Band Power": low_band_power,
            "Mid Band Power": mid_band_power,
            "High Band Power": high_band_power
        }
        task = f'Task {self.tasknumber}' if self.tasknumber != 0 else "Baseline"

        power_spectrum_df = pd.DataFrame(power_spectrum_data)
        # Seleziona solo la prima riga delle colonne Low Band Power, Mid Band Power e High Band Power
        #power_spectrum_df = power_spectrum_df.iloc[0][["Low Band Power", Mid Band Power", "High Band Power"]]
        power_spectrum_df.to_csv(os.path.join(self.output_directory_freq, f"GSR_{task}_P_{self.id}.csv"), index=False)
 
        return low_band_power, mid_band_power, high_band_power, positive_freqs, psd   
    
    def findPeaks(self):
        '''
        Find maximum and minimum peaks in the GSR signal
        '''
        
        peaks, _ = find_peaks(self.df['GSR_phasic'], distance=256) # Returns the indices of the peaks
        peaks_min, _ = find_peaks(1/self.df['GSR_phasic'], distance=256)  # Returns the indices of the peaks (min peaks) on the inverted signal

        return peaks_min, peaks

    def cleanPeaks(self, peaks_min, peaks):
        '''
        Clean the extracted peaks from the GSR signal, by removing two conesecutive 
        peaks of the same type keeping the one with the highest value (max) or the lowest value (min)
        '''
        combined_list = [('min', index, self.df.loc[index, 'GSR_phasic']) for index in peaks_min] + [('max', index, self.df.loc[index, 'GSR_phasic']) for index in peaks]
        sorted_combined_list = sorted(combined_list, key=lambda x: x[1])
        
        sorted_combined_list = [elem for elem in sorted_combined_list if not isnan(elem[2])]
                            
        lastExtreme = ""
        
        for c, entry in enumerate(sorted_combined_list):
            type_extreme, position, value = entry

            if lastExtreme == "":
                lastExtreme = type_extreme
                continue
            
            elif type_extreme == lastExtreme: 
                if type_extreme == "max":
                    if value > sorted_combined_list[c-1][2]:
                        sorted_combined_list[c-1] = sorted_combined_list[c]
                    else:
                        sorted_combined_list[c] = sorted_combined_list[c-1]
                elif type_extreme == "min":
                    if value < sorted_combined_list[c-1][2]:
                        sorted_combined_list[c-1] = sorted_combined_list[c]
                    else:
                        sorted_combined_list[c] = sorted_combined_list[c-1]
                lastExtreme = type_extreme
            else:
                lastExtreme = type_extreme

        # il doppio cast serve a ottenere un oggetto list dalla map, un set per togliere i duplicati e nuovamente una lista per ordinarla
        max_positions = list(set(list(map(lambda x: x[1], filter(lambda x: x[0] == 'max', sorted_combined_list)))))

        min_positions = list(set(list(map(lambda x: x[1], filter(lambda x: x[0] == 'min', sorted_combined_list)))))
        

        return sorted(min_positions), sorted(max_positions)
    def buildNewDF(self, peaks_min, peaks):
        '''
        Build a new DataFrame
        '''
        start_time = self.df["Timestamp"].iloc[0]
        peaks_pos = peaks * 1000 + start_time
        num_peaks = len(peaks)
        peaks_min_pos = peaks_min * 1000 + start_time
        peaks_y = self.df['GSR_phasic'].iloc[peaks].values
        self.new_df = pd.DataFrame({
            'Peaks_Pos': pd.Series(peaks_pos),
            'Peaks_Pos_Inverse': pd.Series(peaks_min_pos),
            'Amplitude': pd.Series(peaks_y)
        }).dropna()

        latency = np.diff(self.new_df['Peaks_Pos_Inverse'])
        rise_time = (self.new_df['Peaks_Pos'] - self.new_df['Peaks_Pos_Inverse']).abs()[:-1]
        recovery_time = latency - rise_time

        self.new_df['Rise_Time'] = rise_time
        self.new_df['Recovery_Time'] = recovery_time
        self.new_df['Latency'] = np.append(latency, 0)
        self.new_df['Peaks Number'] = num_peaks
    
        self.new_df = self.new_df.round(5)
 
        self.new_df = self.new_df.round({'Latency': 3, 'Rise_Time': 3, 'Recovery_Time': 3})
        self.new_df['Latency_Norm'] = round(self.new_df['Latency'] / self.new_df['Latency'].max(), 2)
        self.new_df['Rise_Time_Norm'] = round(self.new_df['Rise_Time'] / self.new_df['Rise_Time'].max(), 2)
        self.new_df['Recovery_Time_Norm'] = round(self.new_df['Recovery_Time'] / self.new_df['Recovery_Time'].max(), 2)
        self.new_df['Amplitude_Norm'] = self.new_df['Amplitude'] / self.new_df['Amplitude'].max()
    
    def windowsDF(self, windowsSize=10):
        self.new_df['Window'] = (self.new_df['Peaks_Pos'] // (windowsSize * 1000)).astype(int)
        aggregations = {
            'Latency': ['mean', 'std', 'min', 'max', 'var', 'median'],
            'Rise_Time': ['mean', 'std', 'min', 'max', 'var', 'median'],
            'Recovery_Time': ['mean', 'std', 'min', 'max', 'var', 'median'],
            'Amplitude': ['mean', 'std', 'min', 'max', 'var', 'median'],
            'Latency_Norm': ['mean', 'min', 'max', 'median'],
            'Rise_Time_Norm': ['mean', 'min', 'max', 'median'],
            'Recovery_Time_Norm': ['mean', 'min', 'max', 'median'],
            'Amplitude_Norm': ['mean', 'min', 'max', 'median']        
            }
        self.reduced_df = self.new_df.groupby('Window').agg(aggregations).reset_index()
        self.reduced_df.columns = ['_'.join(col).strip() for col in self.reduced_df.columns.values]
        self.reduced_df['Window Start'] = self.reduced_df['Window_'] * windowsSize * 1000
        self.reduced_df = self.reduced_df[['Window Start'] + [col for col in self.reduced_df.columns if col != 'Window Start' and col != 'Window_']]

        task = f'Task {self.tasknumber}' if self.tasknumber != 0 else "Baseline"
        self.reduced_df = self.reduced_df.abs()

        metrics = [
            "Rise_Time_mean", "Rise_Time_std", "Rise_Time_min", "Rise_Time_max", "Rise_Time_median",
            "Recovery_Time_mean", "Recovery_Time_std", "Recovery_Time_min", "Recovery_Time_max", "Recovery_Time_median",
            "Amplitude_mean", "Amplitude_std", "Amplitude_min", "Amplitude_max", "Amplitude_median",
        ]

        metrics_rename = [
            "Rise_Time Mean", "Rise_Time Std", "Rise_Time Min", "Rise_Time Max", "Rise_Time Median",
            "Recovery_Time Mean", "Recovery_Time Std", "Recovery_Time Min", "Recovery_Time Max", "Recovery_Time Median",
            "Amplitude Mean", "Amplitude Std", "Amplitude Min", "Amplitude Max", "Amplitude Median"]

        self.reduced_df = self.reduced_df.rename(columns=dict(zip(metrics, metrics_rename)))

        outPath = os.path.join(self.output_directory_windows, f'GSR_{task}_P_{self.id}.csv')
        self.reduced_df.to_csv(outPath, index=False)

    
    def exportCSV(self):
        task = f'Task {self.tasknumber}' if self.tasknumber != 0 else "Baseline"
        outPath = os.path.join(self.output_directory, f'GSR_{task}_P_{self.id}.csv')
        self.new_df.to_csv(outPath, index=False)
        
    def pre_process(self):
        fs = 512
        self.loadData()
        t_l, t_up = self.getThresholds()
        self.removeArtifacts(t_l, t_up)
        self.filterBandPass(fs)
        self.filterWaveletTrasform()
        self.getTonicPhasic()
        self.power_spectrum_analysis(fs)
        peaks_min, peaks = self.findPeaks()
        peaks_min, peaks = self.cleanPeaks(peaks_min, peaks)
        self.buildNewDF(peaks_min, peaks)
        self.exportCSV()
        self.windowsDF(self.window_size)

import os
import pandas as pd
import numpy as np
from scipy.signal import filtfilt, find_peaks, butter
from setup import setup, getConfigData
from Analysis.main_class import Participant
from scipy.stats import entropy
from scipy import signal


class Participant_ECG(Participant):
    
    def __init__(self, filepath):
        super().__init__(filepath)
        config_data = getConfigData()
        setup(config_data)
        self.output_directory = f"{config_data['PFOLDER']['ECG_PROCESSED']}"
        self.output_directory_windows = f"{config_data['PFOLDER']['ECG_PROCESSED_WINDOWS']}"
        self.column_names = []
    
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
                                      dtype={'ECG LA-RA CAL': float, 'ECG LL-RA CAL': float, 'ECG Vx-RL CAL': float},
                                      usecols=range(4))
                
            else:
                self.df = pd.read_csv(self.filepath, 
                                      sep=',', 
                                      dtype={'ECG LA-RA CAL': float, 'ECG LL-RA CAL': float, 'ECG Vx-RL CAL': float,'SourceStimuliName':str})
                valid_source_names = ['60 Sec Video', '60 Sec Video-1', '60 Sec Video-2']
                self.df = self.df[self.df['SourceStimuliName'].isin(valid_source_names)]
                self.df = self.df.drop(['SourceStimuliName'], axis=1)

        except Exception as e:
            raise Exception(f'While loading the data this Exception occurred:\n{e} - {type(e)}')
        try:
            self.df['Timestamp'] = pd.to_datetime(self.df['Timestamp'], format='%Y-%m-%d %H:%M:%S.%f')
        except ValueError:
            self.df['Timestamp'] = pd.to_datetime(self.df['Timestamp'], format='%M:%S.%f')
        
        start_time = self.df['Timestamp'].iloc[0]
        self.df['Timestamp'] = (self.df['Timestamp'] - start_time).dt.total_seconds() * 1000
        self.df.set_index('Timestamp', inplace=True)  # Set Timestamp as index
        
        if 'Index' in self.df.columns:
            self.df.drop(columns=['Index'], inplace=True)  # Drop the default integer index column if it exists
        
        if not len(self.column_names):
            self.column_names = self.df.columns.tolist()
    
    
    def delExtremeNSec(self,n):
        '''
        Delete the first and the last N seconds; in the case of the Baseline task just the first N seconds will be cut. 
        '''
        if self.tasknumber != 0:
            self.df = self.df[(self.df.index > n*1000) & (self.df.index < (self.df.index[-1] - n*1000))]
        else:
            self.df = self.df[(self.df.index > n*1000)]
    
    def delNullInf(self):
        '''
        Delete null and inf. This method avoids having infinite vals and nan vals. 
        '''
        self.df = self.df.replace([np.inf, -np.inf], np.nan)
        self.df = self.df.dropna()
  
    def normalize(self, p_low, p_high, f_low, f_high, column_name, fs=512):
        '''
        Normalize and filter data
        '''
        ecg_data = self.df[column_name]
        p10, p90 = np.percentile(ecg_data, [p_low,p_high])
        ecg_data_norm = (ecg_data - p10) / (p90 - p10)
        b, a = butter(2, [f_low/(fs/2), f_high/(fs/2)], btype='bandpass')
        filter_data = filtfilt(b, a, ecg_data_norm)
        return filter_data


    def calculate_entropy(self, signal, num_bins=10):
        '''
        Calculate the Shannon Entropy of a signal
        '''
        signal = signal[~np.isnan(signal)]
        
        if len(signal) == 0:
            return np.nan
        
        hist, _ = np.histogram(signal, bins=num_bins, density=True)
        hist = hist[hist > 0]  
        entropy = -np.sum(hist * np.log(hist))
        return entropy
    
    def higuchi_fractal_dimension(self, signal, max_k=20):
        '''
        Calculate the Higuchi Fractal Dimension of a signal
        '''
        N = len(signal)
        Lk = []

        for k in range(1, max_k + 1):
            L = 0.0
            for m in range(k):
                L_temp = 0.0
                for j in range(m, N - k, k):
                    L_temp += abs(signal[j + k] - signal[j])
                L_temp = (L_temp * (N - 1)) / ((N - m) * k)
                L += L_temp

            Lk.append(L / k)

        Lk = np.array(Lk)

        if np.any(Lk <= 0):
            raise ValueError('Lk contains non-positive values. Check the signal and the calculation.')

        log_k = np.log(np.arange(1, max_k + 1))
        log_Lk = np.log(Lk)

        coeffs = np.polyfit(log_k[1:], log_Lk[1:], 1)
        coeffs = np.abs(coeffs)
        return coeffs[0]

    def findPeaks(self, n, f, filtered_data): 
        '''
        Extract RR peaks and add other metrics to the dataframe
        '''
        self.df['Index'] = range(1, len(self.df) + 1)
        self.df.set_index('Index')
        peaks, _ = find_peaks(filtered_data, height=-100, distance=200)
        filtered_data_series = pd.Series(filtered_data)

        filtered_data_mean = filtered_data_series.rolling(n).mean()
        filtered_data_mean = filtered_data_mean + f * filtered_data_series.rolling(n).std()
        filtered_data_mean = filtered_data_mean.bfill()

        valid_peaks = peaks[filtered_data_series[peaks] > filtered_data_mean[peaks]]

        peaks_pos_array = self.df.index[valid_peaks]
    
        rr_intervals = np.diff(peaks_pos_array)/(1000/512)

        heart_rate = 60000 / rr_intervals

        rr_intervals = np.around(rr_intervals, decimals=0)
        heart_rate = np.around(heart_rate, decimals=0)
        
        self.new_df = pd.DataFrame({'RR_Intervals': rr_intervals, 'Heart_Rate': heart_rate})
        numPeakdf = pd.DataFrame({'Peaks_pos': peaks_pos_array})
        
        self.new_df = pd.concat([numPeakdf, self.new_df], axis=1)
        self.new_df.dropna(inplace=True)
        
        # Add Higuchi FD and Shannon Entropy 
        self.new_df['Higuchi_FD'] = self.higuchi_fractal_dimension(filtered_data)
        self.new_df['Shannon_Entropy'] = self.calculate_entropy(filtered_data)
        
        self.new_df.loc[1:, 'Higuchi_FD'] = np.nan
        self.new_df.loc[1:, 'Shannon_Entropy'] = np.nan

        return peaks, valid_peaks, filtered_data_mean, filtered_data_series
    
    def cleanHeartRate(self, coeff=1.5):
        '''
        Filtering values using the multiplication of a coefficient by the standard deviation to define thresholds.
        '''
        threshold_upper = round(self.new_df['Heart_Rate'].mean(), 0) + coeff * round(self.new_df['Heart_Rate'].std(), 0) 
        threshold_lower = round(self.new_df['Heart_Rate'].mean(), 0) - coeff * round(self.new_df['Heart_Rate'].std(), 0)
        
        self.new_df.loc[(self.new_df['Heart_Rate'] < threshold_lower) | (self.new_df['Heart_Rate'] > threshold_upper), 'Heart_Rate'] = np.nan

        self.new_df['Heart_Rate'] = self.new_df['Heart_Rate'].ffill()
        
    def addStats(self):
        '''
        Add more normalized metrics and some related to the HRV. 
        '''
        max_rr = self.new_df['RR_Intervals'].max()
        max_hr =  self.new_df['Heart_Rate'].max()

        self.new_df['RR_norm'] =  round(self.new_df['RR_Intervals'] / max_rr, 2)
        self.new_df['HR_norm'] =  round(self.new_df['Heart_Rate'] / max_hr,2)
        diff_rr = self.new_df['RR_Intervals'].diff().abs() ** 2
        
        rmssd = round(np.sqrt((diff_rr.sum())/(len(diff_rr)-1)), 3)
        nn50_count = np.sum(diff_rr > 50)
        nn25_count = np.sum(diff_rr > 25)
        pnn50 = round((nn50_count / (len(diff_rr) - 1)) * 100, 3) # Subtract 1 because there are n-1 differences between n intervals
        pnn25 = round((nn25_count / (len(diff_rr) - 1)) * 100, 3) 
        sdnn = round(self.new_df['RR_Intervals'].std(), 3)
        
        self.new_df['PNN25'] = np.nan
        self.new_df['SDNN'] = np.nan
        self.new_df['RMSSD'] = np.nan
        self.new_df['PNN50'] = np.nan

        self.new_df.at[0, 'PNN25'] = pnn25
        self.new_df.at[0, 'SDNN'] = sdnn
        self.new_df.at[0, 'RMSSD'] = rmssd
        self.new_df.at[0, 'PNN50'] = pnn50
        
        # Move Higuchi_FD and Shannon_Entropy columns to the end
        cols = list(self.new_df.columns)
        cols.append(cols.pop(cols.index('Higuchi_FD')))
        cols.append(cols.pop(cols.index('Shannon_Entropy')))
        self.new_df = self.new_df[cols]
    
    def addFrequencyBandPower(self):
        '''
        Add the power of the frequency bands VLF, LF, and HF.
        '''
        band_powers = {
            'vlf' : [0.01, 0.04],
            'lf' : [0.04, 0.15],
            'hf' : [0.15, 0.4]
        }
        fs_HR = 1000 / self.new_df['RR_Intervals'].mean()
        hr = self.new_df['Heart_Rate']
        # Compute the power spectral density of the HR signal
        f, Pxx = signal.periodogram(hr, fs=fs_HR, scaling='spectrum')

        vlf_indices = np.where(np.logical_and(f >= band_powers['vlf'][0], 
                                                f <= band_powers['vlf'][1]))
        
        lf_indices = np.where(np.logical_and(f >= band_powers['lf'][0], 
                                                f <= band_powers['lf'][1]))
        
        hf_indices = np.where(np.logical_and(f >= band_powers['hf'][0], 
                                                f <= band_powers['hf'][1]))
        Pxx = Pxx[~np.isnan(Pxx)]
        # Compute the power spectrum for each frequency band, if the indices are not empty
        try:
            vlf_power = np.mean(Pxx[vlf_indices])
            lf_power = np.mean(Pxx[lf_indices])
            hf_power = np.mean(Pxx[hf_indices]) 
            
            self.new_df['VLF'] = np.nan
            self.new_df['LF'] = np.nan
            self.new_df['HF'] = np.nan
            
            self.new_df.iloc[0, self.new_df.columns.get_loc('VLF')] = vlf_power
            self.new_df.iloc[0, self.new_df.columns.get_loc('LF')] = lf_power
            self.new_df.iloc[0, self.new_df.columns.get_loc('HF')] = hf_power
        except IndexError:
            self.new_df['VLF'] = np.nan
            self.new_df['LF'] = np.nan
            self.new_df['HF'] = np.nan
        
        
    def saveDataframe(self):
        task = f'Task_0{self.tasknumber}' if self.tasknumber != 0 else 'Baseline'
        ECG_path = os.path.join(self.output_directory, f'./ECG_{task}_P_{self.id}_Heart_Parameters.csv')

        self.new_df.to_csv(ECG_path, index=False)
    
    def divideInWindows(self, wDim):
        '''
        Divide the data in windows of wDim seconds and calculate the statistics for each window.
        '''
        num_windows = int(self.new_df['Peaks_pos'].max() / (wDim * 1000)) + 1
        
        window_starts = []
        
        #-----RR-----
        rr_intervals_means = []
        rr_intervals_means_norm = []
        rr_intervals_min_norm = []
        rr_intervals_max_norm = []
        rr_intervals_median_norm = []
        rr_intervals_min = []
        rr_intervals_max = []
        rr_intervals_median = []
        
        #-----HR-----
        heart_rate_means = []
        heart_rate_means_norm = []
        heart_rate_min_norm = []
        heart_rate_max_norm = []
        heart_rate_median_norm = []
        heart_rate_std = []
        heart_rate_min = []
        heart_rate_max = []
        heart_rate_var = []
        heart_rate_median = []
        
        
        #-----PNN25,PNN50,RMSDD,SDNN-----
        pnn25_list = []
        pnn50_list = []
        rmssd_list = []
        sdnn_list = []
        
        '''
        #-----Higuchi FD, Shannon Entropy-----
        higuchi_fd_list = []
        shannon_entropy_list = []
        '''
        
        # Iterate over the windows
        for i in range(num_windows):
            window_start = i * wDim * 1000
            window_end = window_start + wDim * 1000
            window_data = self.new_df[(self.new_df['Peaks_pos'] >= window_start) & (self.new_df['Peaks_pos'] < window_end)]
            if len(window_data) > 0:
                window_starts.append(window_start)
                #------Value RR-------
                rr_intervals_means.append(round(window_data['RR_Intervals'].mean(), 2))
                rr_intervals_min.append(round(window_data['RR_Intervals'].min(), 2))
                rr_intervals_max.append(round(window_data['RR_Intervals'].max(), 2))
                rr_intervals_median.append(round(window_data['RR_Intervals'].median(), 2))

                #-------Value HR-------
                heart_rate_means.append(round(window_data['Heart_Rate'].mean(), 2))
                heart_rate_std.append(round(window_data['Heart_Rate'].std(), 2))
                heart_rate_min.append(round(window_data['Heart_Rate'].min(), 2))
                heart_rate_max.append(round(window_data['Heart_Rate'].max(), 2))
                heart_rate_var.append(round(window_data['Heart_Rate'].var(), 2))
                heart_rate_median.append(round(window_data['Heart_Rate'].median(), 2))

                #----Value RR Norm -----
                rr_intervals_means_norm.append(round(window_data['RR_norm'].mean(), 2))
                rr_intervals_min_norm.append(round(window_data['RR_norm'].min(), 2))
                rr_intervals_max_norm.append(round(window_data['RR_norm'].max(), 2))
                rr_intervals_median_norm.append(round(window_data['RR_norm'].median(), 2))

                #-------Value HR norm-------
                heart_rate_means_norm.append(round(window_data['HR_norm'].mean(), 2))
                heart_rate_min_norm.append(round(window_data['HR_norm'].min(), 2))
                heart_rate_max_norm.append(round(window_data['HR_norm'].max(), 2))
                heart_rate_median_norm.append(round(window_data['HR_norm'].median(), 2))
                
                #-----PNN25,PNN50,RMSDD,SDNN-----
                diff_rr = window_data['RR_Intervals'].diff().abs() ** 2
                nn50_count = np.sum(diff_rr > 50)
                nn25_count = np.sum(diff_rr > 25)
                
                if len(diff_rr) > 1:
                    rmssd = round(np.sqrt((diff_rr.sum()) / (len(diff_rr) - 1)), 3)
                    pnn50 = round((nn50_count / (len(diff_rr) - 1)) * 100, 3)
                    pnn25 = round((nn25_count / (len(diff_rr) - 1)) * 100, 3)
                    sdnn = round(window_data['RR_Intervals'].std(), 3)
                else:
                    rmssd = np.nan
                    pnn50 = np.nan
                    pnn25 = np.nan
                    sdnn = np.nan

                
                pnn25_list.append(pnn25)
                pnn50_list.append(pnn50)
                rmssd_list.append(rmssd)
                sdnn_list.append(sdnn)
                

        result_df = pd.DataFrame({
            'Window Start': window_starts,
            #-----Value for RR-----
            'RR_Intervals_Mean': rr_intervals_means,
            'RR_Intervals_Min': rr_intervals_min,
            'RR_Intervals_Max': rr_intervals_max,
            'RR_Intervals_Median': rr_intervals_median,
            #----Value for HR-----
            'Heart_Rate_Mean': heart_rate_means,
            'Heart_Rate_Min': heart_rate_min,
            'Heart_Rate_Max': heart_rate_max,
            'Heart_Rate_Std': heart_rate_std,
            'Heart_Rate_Var': heart_rate_var,
            'Heart_Rate_Median': heart_rate_median,
            #----Value for RR normalized----
            'RR_norm_Mean': rr_intervals_means_norm,
            'RR_norm_Min': rr_intervals_min_norm,
            'RR_norm_Max': rr_intervals_max_norm,
            'RR_norm_Median': rr_intervals_median_norm,
            #----Value for HR normalized----
            'HR_norm_Mean': heart_rate_means_norm,
            'HR_norm_Min': heart_rate_min_norm,
            'HR_norm_Max': heart_rate_max_norm,
            'HR_norm_Median': heart_rate_median_norm,
            #----General values------
            'PNN25': pnn25_list,
            'SDNN': sdnn_list,            
            'RMSSD': rmssd_list,
            'PNN50': pnn50_list,
        })
        
        #----Higuchi FD, Shannon Entropy----
        result_df['Higuchi_FD'] = np.nan
        result_df['Shannon_Entropy'] = np.nan
        
        result_df.at[0, 'Higuchi_FD'] = self.new_df['Higuchi_FD'].iloc[0]
        result_df.at[0, 'Shannon_Entropy'] = self.new_df['Shannon_Entropy'].iloc[0]
        
        #----Frequency Band Power----
        result_df['VLF'] = np.nan
        result_df['LF'] = np.nan
        result_df['HF'] = np.nan
        
        result_df.at[0, 'VLF'] = self.new_df['VLF'].iloc[0]
        result_df.at[0, 'LF'] = self.new_df['LF'].iloc[0]
        result_df.at[0, 'HF'] = self.new_df['HF'].iloc[0]
        
                    
        
        task = f'Task_0{self.tasknumber}' if self.tasknumber != 0 else "Baseline"
                
        ECG_path = os.path.join(self.output_directory_windows, f'./ECG_{task}_P_{self.id}_Heart_Parameters.csv')
        result_df.to_csv(ECG_path, index=False)
    
    def pre_process(self):
        """
        Apply pre-process pipeline to the istance of the patient
        """
        self.loadData()
        if self.id > 11:
            self.delExtremeNSec(5)
        self.delNullInf()
        filtered_data = self.normalize(10, 90, 10, 40, "ECG LL-RA CAL")
        f = 1
        peaks, valid_peaks, filtered_data_mean, filtered_data_series = self.findPeaks(1000,f, filtered_data)
        self.addStats()
        self.cleanHeartRate()
        self.addFrequencyBandPower()
        self.saveDataframe()
        self.divideInWindows(self.window_size)
                      


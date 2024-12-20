import os
import re
import mne
import pandas as pd
import numpy as np
import json
from setup import setup, getConfigData
import traceback
from scipy.stats import kurtosis
from mne.preprocessing import ICA
from scipy import signal
from Analysis.main_class import Participant
from mne.time_frequency import tfr_array_morlet

from scipy.signal import coherence

class Participant_EEG(Participant):
    def __init__(self, filepath):
        super().__init__(filepath)
        config_data = getConfigData()
        setup(config_data)
        self.output_directory_p = f"{config_data['PFOLDER']['EEG_CSV']}"
        self.output_directory_fif = f"{config_data['PFOLDER']['EEG_FIF']}"
        self.output_directory_reduced = f"{config_data['PFOLDER']['EEG_REDUCED']}"
        
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
                                    usecols=range(21)
                                    )
            else:
                self.df = pd.read_csv(self.filepath, 
                                    sep=',',
                                    dtype={'SourceStimuliName': str}
                                    )
                valid_source_names = ["60 Sec Video", "60 Sec Video-1", "60 Sec Video-2"]
                self.df = self.df[self.df['SourceStimuliName'].isin(valid_source_names)]
                self.df = self.df.drop(['SourceStimuliName'], axis=1)

        except Exception as e:
            raise Exception(f"While loading the data this Expection occurred:\n{e} - {type(e)}")
        
        self.columns = self.df.columns.tolist()[1:]
          
        try:
            self.df['Timestamp'] = pd.to_datetime(self.df['Timestamp'], format="%Y-%m-%d %H:%M:%S.%f")
            self.df['Timestamp'] = (self.df['Timestamp'] - self.df['Timestamp'].min()).astype(np.int64) // 10**6
            self.df['Timestamp'] = self.df['Timestamp'].astype(int)
        except ValueError:
            self.df['Timestamp'] = self.df.index * 1/500

    def filterData(self):
        """
		Downsampling of the channels, removing channels related to eyes, pass band filter
		"""	
        df = self.df[self.columns].to_numpy()

        info = mne.create_info(ch_names=self.columns, sfreq=500, ch_types='misc')
        self.raw = mne.io.RawArray(df.T, info, verbose=False)

        type_dict = {ch: 'eeg' if ch not in ['EOG1', 'EOG2'] else 'eog' for ch in self.columns}
        del type_dict['EXT']
        self.raw.set_channel_types(type_dict, verbose=False)

        self.raw.crop(tmin=5, tmax=self.raw.times[-1] - 5)

        self.raw.resample(250)

        self.eog = self.raw.copy().pick(['EOG1', 'EOG2', 'EXT'])
        self.raw.drop_channels(['EOG1', 'EOG2', 'EXT'])
        channels = self.raw.ch_names
        self.raw.filter(1, 30, picks=channels, verbose=False)
        mne.baseline.rescale(self.raw._data, self.raw.times, baseline=(None, 0), verbose=False)
    
    def removeBadCh(self):
        """ 
        Bad channels removal with Kurtosis method
        """
        
        kurt = kurtosis(self.raw.get_data(), axis=1)
        bad_channels = np.where(kurt > 5)[0]
        self.raw.info['bads'] = [self.raw.ch_names[i] for i in bad_channels]

        bad_channels = self.raw.info['bads']

        if bad_channels:
            try:
                self.raw.drop_channels(bad_channels)
            except ValueError as e:
                print(f"Error {traceback.format_exc()}")

        n_channels = len(self.raw.ch_names)

        if n_channels > 0:
            channels = self.raw.ch_names
            ica = ICA(n_components=n_channels, random_state=97, verbose=False)
            ica.fit(self.raw, picks=channels, verbose=False)
            ica.apply(self.raw, verbose=False)
        else:
            raise ValueError("All channells are bad.")

    def get_power_spectrum(self, window_length):
        channels = self.raw.ch_names
        power_dict = {f"{key}_{band}_mean": [] for key in channels for band in ['theta', 'alpha', 'beta']}
        coherence_dict = {f"{key1}_{key2}_coherence": [] for key1 in channels for key2 in channels if key1 != key2}
        min_dict = {f"{channel_band.replace('_mean','')}_min": [] for channel_band in power_dict.keys()}
        max_dict = {f"{channel_band.replace('_mean','')}_max": [] for channel_band in power_dict.keys()}
        median_dict = {f"{channel_band.replace('_mean','')}_median": [] for channel_band in power_dict.keys()}
        std_dict = {f"{channel_band.replace('_mean','')}_std": [] for channel_band in power_dict.keys()}
        beta_alpha_ratio = {}
        assymetry_dict = {}

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
                avg_power_mean = np.mean(avg_power, axis=1)
                avg_power_min = np.min(avg_power, axis=1)
                avg_power_max = np.max(avg_power, axis=1)
                avg_power_median = np.median(avg_power, axis=1)
                avg_power_std = np.std(avg_power, axis=1)
                for j, channel in enumerate(channels):
                    power_dict[f'{channel}_{band_name}_mean'].append(avg_power_mean[j])
                    min_dict[f'{channel}_{band_name}_min'].append(avg_power_min[j])
                    max_dict[f'{channel}_{band_name}_max'].append(avg_power_max[j])
                    median_dict[f'{channel}_{band_name}_median'].append(avg_power_median[j])
                    std_dict[f'{channel}_{band_name}_std'].append(avg_power_std[j])

            # Calculate coherence between pairs of channels
            for j in range(len(channels)):
                for k in range(j + 1, len(channels)):
                    f, Cxy = coherence(data_window[j], data_window[k], fs=self.raw.info['sfreq'])
                    avg_coherence = np.mean(Cxy) if len(Cxy) > 0 else np.nan  # Handle empty results
                    coherence_dict[f'{channels[j]}_{channels[k]}_coherence'].append(avg_coherence)

            for channel in power_dict.keys(): 
                if "beta" in channel:
                    beta_channel_name = channel.replace("_mean", "")
                    alpha_channel_name = channel.replace("beta", "alpha").replace("_mean", "")

                    
                    beta_power_array = np.array(power_dict[f"{beta_channel_name}_mean"])
                    alpha_power_array = np.array(power_dict[f"{alpha_channel_name}_mean"])
                    beta_alpha_ratio[f"{beta_channel_name}/{alpha_channel_name}_mean"] = list(np.round(beta_power_array / alpha_power_array, 4))
                    
                    beta_power_array_min = np.array(min_dict[f'{beta_channel_name}_min'])
                    alpha_power_array_min = np.array(min_dict[f'{alpha_channel_name}_min'])
                    beta_alpha_ratio[f"{beta_channel_name}/{alpha_channel_name}_min"] = list(np.round(beta_power_array_min / alpha_power_array_min, 4))
                    
                    beta_power_array_max = np.array(max_dict[f'{beta_channel_name}_max'])
                    alpha_power_array_max = np.array(max_dict[f'{alpha_channel_name}_max'])
                    beta_alpha_ratio[f"{beta_channel_name}/{alpha_channel_name}_max"] = list(np.round(beta_power_array_max / alpha_power_array_max, 4))
                    
                    beta_power_array_median = np.array(median_dict[f'{beta_channel_name}_median'])
                    alpha_power_array_median = np.array(median_dict[f'{alpha_channel_name}_median'])
                    beta_alpha_ratio[f"{beta_channel_name}/{alpha_channel_name}_median"] = list(np.round(beta_power_array_median / alpha_power_array_median, 4))

                    beta_power_array_std = np.array(std_dict[f'{beta_channel_name}_std'])
                    alpha_power_array_std = np.array(std_dict[f'{alpha_channel_name}_std'])
                    beta_alpha_ratio[f"{beta_channel_name}/{alpha_channel_name}_std"] = list(np.round(beta_power_array_std / alpha_power_array_std, 4))
                    
            if "F3" and "F4" in channels:
                f3_alpha_mean = np.array(power_dict['F3_alpha_mean'])
                f4_alpha_mean = np.array(power_dict['F4_alpha_mean'])
                assymetry_dict["F3_F4_alpha_Asymmetry_mean"] = list(np.log(f3_alpha_mean) - np.log(f4_alpha_mean))
                
                f3_alpha_min = np.array(min_dict['F3_alpha_min'])
                f4_alpha_min = np.array(min_dict['F4_alpha_min'])
                assymetry_dict["F3_F4_alpha_Asymmetry_min"] = list(np.log(f3_alpha_min) - np.log(f4_alpha_min))
                
                f3_alpha_max = np.array(max_dict['F3_alpha_max'])
                f4_alpha_max = np.array(max_dict['F4_alpha_max'])
                assymetry_dict["F3_F4_alpha_Asymmetry_max"] = list(np.log(f3_alpha_max) - np.log(f4_alpha_max))
                
                f3_alpha_median = np.array(median_dict['F3_alpha_median'])
                f4_alpha_median = np.array(median_dict['F4_alpha_median'])
                assymetry_dict["F3_F4_alpha_Asymmetry_median"] = list(np.log(f3_alpha_median) - np.log(f4_alpha_median))
                
                f3_alpha_std = np.array(std_dict['F3_alpha_std'])
                f4_alpha_std = np.array(std_dict['F4_alpha_std'])
                assymetry_dict["F3_F4_alpha_Asymmetry_std"] = list(np.log(f3_alpha_std) - np.log(f4_alpha_std))
            
            if "F7" and "F8" in channels: 
                f7_alpha_mean = np.array(power_dict['F7_alpha_mean'])
                f8_alpha_mean = np.array(power_dict['F8_alpha_mean'])
                assymetry_dict["F7_F8_alpha_Asymmetry_mean"] = list(np.log(f7_alpha_mean) - np.log(f8_alpha_mean))
                
                f7_alpha_min = np.array(min_dict['F7_alpha_min'])
                f8_alpha_min = np.array(min_dict['F8_alpha_min'])
                assymetry_dict["F7_F8_alpha_Asymmetry_min"] = list(np.log(f7_alpha_min) - np.log(f8_alpha_min))
                
                f7_alpha_max = np.array(max_dict['F7_alpha_max'])
                f8_alpha_max = np.array(max_dict['F8_alpha_max'])
                assymetry_dict["F7_F8_alpha_Asymmetry_max"] = list(np.log(f7_alpha_max) - np.log(f8_alpha_max))
                
                f7_alpha_median = np.array(median_dict['F7_alpha_median'])
                f8_alpha_median = np.array(median_dict['F8_alpha_median'])
                assymetry_dict["F7_F8_alpha_Asymmetry_median"] = list(np.log(f7_alpha_median) - np.log(f8_alpha_median))
                
                f7_alpha_std = np.array(std_dict['F7_alpha_std'])
                f8_alpha_std = np.array(std_dict['F8_alpha_std'])
                assymetry_dict["F7_F8_alpha_Asymmetry_std"] = list(np.log(f7_alpha_std) - np.log(f8_alpha_std))

        # Ensure all lists are the same length by filling with NaN
        max_length = n_windows
        for key in coherence_dict.keys():
            while len(coherence_dict[key]) < max_length:
                coherence_dict[key].append(np.nan)

        #---------------------------------------
        new_df_power = pd.DataFrame(power_dict)
        min_df = pd.DataFrame(min_dict)
        max_dict = pd.DataFrame(max_dict)
        median_df = pd.DataFrame(median_dict)
        std_df = pd.DataFrame(std_dict)

        new_df_power = pd.concat([new_df_power, min_df, max_dict, median_df, std_df], axis=1)

        new_df_power = new_df_power.reindex(sorted(new_df_power.columns), axis=1)
        #---------------------------------------
        new_df_coherence = pd.DataFrame(coherence_dict)
        new_df_coherence.dropna(axis=1, how='all', inplace=True)
        #---------------------------------------
        beta_alpha_ratio_df = pd.DataFrame(beta_alpha_ratio)
        beta_alpha_ratio_df = beta_alpha_ratio_df.reindex(sorted(beta_alpha_ratio_df.columns), axis=1)
        #---------------------------------------
        assymetry_df = pd.DataFrame(assymetry_dict)
        #---------------------------------------
        

        # Concatenate all dataframes
        dataframe_list = [new_df_power, new_df_coherence, beta_alpha_ratio_df]
        if not assymetry_df.empty:
            dataframe_list.append(assymetry_df)
        
        self.new_df_power = pd.concat(dataframe_list, axis=1)

        # Handle timestamps
        time_stamp = list(range(window_length, window_length * (n_windows + 1), window_length))
        self.new_df_power.insert(0, "Timestamp", time_stamp)


        return power_dict, coherence_dict


    def reduceCSV(self):
        """
        Calculate average for each band
        """
        
        timestamp = self.new_df_power['Timestamp']
        alpha_cols = [col for col in self.new_df_power.columns if col.endswith('_alpha')]
        beta_cols = [col for col in self.new_df_power.columns if col.endswith('_beta')]
        theta_cols = [col for col in self.new_df_power.columns if col.endswith('_theta')]

        alpha_mean = self.new_df_power[alpha_cols].mean(axis=1)
        beta_mean =self.new_df_power[beta_cols].mean(axis=1)
        theta_mean = self.new_df_power[theta_cols].mean(axis=1)

        new_df_reduced = pd.DataFrame({
            'Window Start': timestamp,
            'Alpha_mean': alpha_mean,
            'Beta_mean': beta_mean,
            'Theta_mean': theta_mean,
            'Power_beta_alpha_ratio' : round(beta_mean/alpha_mean,3)
        })
        

        task = f'Task_0{self.tasknumber}' if self.tasknumber != 0 else "Baseline"
        reduced_path = os.path.join(self.output_directory_reduced, f'./EEG_{task}_P_{self.id}_reduced.csv')
        new_df_reduced.to_csv(reduced_path, index=False)
        
    def saveDataframe(self):
        """
        Save the dataframe
        """
        task = f'Task_0{self.tasknumber}' if self.tasknumber != 0 else "Baseline"
        output_power_path = os.path.join(self.output_directory_p, f'EEG_{task}_P_{self.id}.csv')
        self.new_df_power.to_csv(output_power_path, index=False)

    def saveFIF(self):
        """
		Save the FIF file
		"""
        task = f'Task_0{self.tasknumber}' if self.tasknumber != 0 else "Baseline"
        EEG_path = os.path.join(self.output_directory_fif, f'./EEG_{task}_P_{self.id}.fif')
        self.raw.save(EEG_path, overwrite=True, verbose=False)
    
    def addStats(self):
        '''
        Add the statistics to the dataframe
        '''
        pass

    def pre_process(self):
        """
        Apply pre-process pipeline to the istance of the patient
        """
        self.loadData()
        self.filterData()
        self.removeBadCh()
        self.get_power_spectrum(self.window_size)
        self.saveDataframe()
        self.saveFIF()
        self.reduceCSV()
        



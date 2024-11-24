import os
import numpy as np
import pandas as pd
from Analysis.main_class import Participant
from setup import setup, getConfigData

class Participant_EMO(Participant):
    
    def __init__(self, file_path):
        super().__init__(file_path)
        config_data = getConfigData()
        setup(config_data)
        self.output_directory_windows = f"{config_data['PFOLDER']['EMOTIONS_PROCESSED']}"
    
    def loadData(self):
        try:
            if self.tasknumber != 0: 
                self.df = pd.read_csv(self.filepath, sep=',', usecols=range(13))
            else:
                self.df = pd.read_csv(self.filepath, sep=',', usecols=range(14))
                valid_source_names = ["60 Sec Video", "60 Sec Video-1", "60 Sec Video-2"]
                self.df = self.df[self.df['SourceStimuliName'].isin(valid_source_names)]
                self.df = self.df.drop(['SourceStimuliName'], axis=1)
        except Exception as e:
            raise Exception(f"While loading the data this Exception occurred:\n{e} - {type(e)}")
    
    def normalize(self):
        for col in self.df.columns:
            if col != 'Timestamp':
                self.df[col] = self.df[col].div(self.df[col].max()).round(3)
    
    def windows(self, time = 10):
        self.df['Timestamp'] = pd.to_datetime(self.df['Timestamp'])
        self.df.set_index('Timestamp', inplace=True)
        self.df = self.df.resample(f'{time}s').agg(['mean', 'std', 'min', 'max'])
        self.df.reset_index(inplace=True)
        self.df['Window Start'] = (self.df['Timestamp'] - self.df['Timestamp'].min()).astype(np.int64) // 10**6
        self.df.set_index('Window Start', inplace=True)
        self.df.drop('Timestamp', axis=1, level=0, inplace=True)
        self.df.columns = ['_'.join(col).strip() for col in self.df.columns.values]
    

    def rename_columns(self):
        new_columns = {}
        for col in self.df.columns:
            if 'mean' in col:
                new_columns[col] = col.replace('_mean', '-mean')
            elif 'std' in col:
                new_columns[col] = col.replace('_std', '-std')
            elif 'min' in col:
                new_columns[col] = col.replace('_min', '-min')
            elif 'max' in col:
                new_columns[col] = col.replace('_max', '-max')
        self.df.rename(columns=new_columns, inplace=True)
    
    def saveNewDf(self):
        task = f'Task_0{self.tasknumber}' if self.tasknumber != 0 else "Baseline"
        outPath = os.path.join(self.output_directory_windows, f'Emotions_{task}_P_{self.id}_processed.csv')
        self.df.to_csv(outPath, index=True)
        
    
    def pre_process(self):
        self.loadData()
        self.normalize()
        self.windows(10)
        self.rename_columns()
        self.saveNewDf()
      
        
        
    
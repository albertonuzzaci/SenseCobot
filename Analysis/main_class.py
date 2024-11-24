from abc import ABC, abstractmethod
from setup import setup, getConfigData

class Participant(ABC):
    def __init__(self, filepath):
        self.filepath = filepath
        self.id = self.get_id()
        self.tasknumber = self.get_tasknumber()
        config_data = getConfigData()
        setup(config_data)
        self.window_size = config_data["WINDOW_SIZE"]

    @abstractmethod
    def pre_process(self):
        """Abstract method that will launch pre-processing"""
        pass

    def get_tasknumber(self):
        """Get task number from file path"""
        if "Baseline" in self.filepath:
            return 0
        else:
            return int(self.filepath.split("_")[-4][-1])
    
    def get_id(self):
        """Get id number from file path"""
        return int(self.filepath.split("_")[-2])
    
import yaml
import os

def makeDir(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

def getConfigData():
    with open('config.yaml','r') as file:
        config_data = yaml.safe_load(file)
    return config_data

def setup(config_data): 
    for k,v in config_data['PFOLDER'].items():
        makeDir(f"{config_data['PFOLDER'][k]}")
    makeDir(f"{config_data['FINAL_DATASET']['MAIN_DIR']}")


if __name__ == "__main__":
	setup(getConfigData())
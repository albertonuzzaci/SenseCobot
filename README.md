# Sense Cobot

## Introduction
This project is designed to process and analyze various physiological signals such as ECG, EEG, GSR, and EMO. The project includes scripts for pre-processing, merging datasets, and generating labels.

## Table of Contents
1. [Virtual Environment Setup](#virtual-environment-setup)
2. [General Setup](#installation)
3. [Usage](#usage)
4. [Contributing](#contributing)

## Genereal Setup

### Dataset

The dataset is available on [Zenodo](https://zenodo.org/), specifically at the following link:
[https://zenodo.org/records/10124005](https://zenodo.org/records/10124005)

### Configuration

The `config.yaml` file is a configuration file that allows you to set the input and output folders.

Once the downloaded folder is renamed to "Dataset", the input folders are standard. The output folders can be customized as desired.


## Virtual Environment Setup
To ensure that all dependencies are managed properly, it is recommended to use a virtual environment. Follow the steps below to activate the virtual environment:
### Windows
1. Open a terminal and navigate to the project directory.
2. Run the following command to activate the virtual environment:
	```sh
	.\venv\Scripts\activate
	```
3. Install the required dependencies:
	```sh
	pip install -r requirements.txt
	```

### macOS/Linux
1. Open a terminal and navigate to the project directory.
2. Run the following command to activate the virtual environment:
	```sh
	source venv/bin/activate
	```
3. Install the required dependencies:
	```sh
	pip install -r requirements.txt
	```
## Usage

### Pre-process
To pre-process the physiological signals, use the following command:
```sh
python3 pre_process.py [signals ...]
```
This script processes the specified signals. If no signals are specified, all signals (ECG, EEG, Emotions, and GSR) are processed. For example:
- To process all signals:
	```sh
	python3 pre_process.py
	```
- To process only ECG and EEG:
	```sh
	python3 pre_process.py ECG EEG
	```

### Merging
After analyzing the data, you can merge the datasets into unique datasets for each signal and finally into one unified dataset using the `merge_dataset` script.
```sh
python3 merge_dataset.py
```

### Contributors

- **Alberto Nuzzaci**
- **Simone Borghi** 

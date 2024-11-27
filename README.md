# Sense Cobot

## Introduction
This project is designed to process and analyze various physiological signals such as ECG, EEG, GSR, and EMO. The project includes scripts for pre-processing, merging datasets, and generating labels.

## Table of Contents
1. [Virtual Environment Setup](#virtual-environment-setup)
2. [General Setup](#general-setup)
3. [Usage](#usage)
4. [Contributors](#contributors)

## General Setup

### Dataset

The dataset is available on [Zenodo](https://zenodo.org/), specifically at the following link:
[https://zenodo.org/records/10124005](https://zenodo.org/records/10124005)

### Configuration

The `config.yaml` file is a configuration file that allows you to set the input and output folders.

Once the downloaded folder is renamed to "Dataset", the input folders are standard. The output folders can be customized as desired.


#### Configuration settings for parallelizing the preprocessing of specific signals.

The `config.yaml` file contains also a dictionary `PARALLELIZE` used to determine whether to parallelize the preprocessing of each specific signal type. 
You can customize whether to enable parallelization for each signal type (ECG, EEG, GSR, EMOTIONS) by setting the corresponding value to `True` or `False`.

It is recommended to leave these settings as they are.

Modify the number of workers according to your hardware specifications. It is recommended to set the number of workers equal to the number of CPU cores available on your machine.



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
Note that only the signals that have been pre-processed will be merged.

### Errors
Errors that occur during pre-processing are not printed to the screen but are saved in an auto-generated file in the current folder called `error.json`.

## Contributors

- **Alberto Nuzzaci**
- **Simone Borghi** 

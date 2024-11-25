import os
import warnings
from sklearn.exceptions import ConvergenceWarning
import json
from Analysis.ECG import Participant_ECG
from Analysis.EEG import Participant_EEG
from Analysis.GSR import Participant_GSR
from Analysis.EMO import Participant_EMO
import contextlib
import io
from setup import setup, getConfigData
import traceback
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager
import argparse

config_data = getConfigData()
setup(config_data)

inputs = {
    "ECG": f"{config_data['FOLDER']['ECG']}",
    "EEG": f"{config_data['FOLDER']['EEG']}",
    "GSR": f"{config_data['FOLDER']['GSR']}",
    "EMO": f"{config_data['FOLDER']['EMOTIONS']}"
}

parallelize = {
    "ECG": False,
    "EEG": True,
    "GSR": True,
    "EMO": True
}

def create_participant(signal, file_path):
    class_name = f"Participant_{signal}"
    if class_name in globals():
        return globals()[class_name](file_path)
    else:
        raise ValueError(f"Class '{class_name}' not found")

def preprocess_file(file_name, input_dir, signal, errors):
    if not file_name.endswith(".csv"):
        return  

    print(f"Preprocessing {file_name}...")
    p = create_participant(signal, os.path.join(input_dir, file_name))

    try: 
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            with contextlib.redirect_stdout(io.StringIO()):  
                p.pre_process()
        print(f"...preprocessing {signal} {file_name} completed!")
    except Exception as e:
        errors[signal].append((file_name, traceback.format_exc()))
        print(traceback.format_exc())
        print(f"...preprocessing {signal} {file_name} raised an ERROR!")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some signals.")
    parser.add_argument('signals', nargs='*', default=['ECG', 'EEG', 'GSR', 'EMO'], help="Signals to process (default: all signals)")

    args = parser.parse_args()
    selected_signals = args.signals

    manager = Manager()
    errors = manager.dict({signal: manager.list() for signal in selected_signals})

    try:
        for signal in selected_signals:
            if signal not in inputs:
                print(f"Signal {signal} is not recognized. Skipping...")
                continue

            input_dir = inputs[signal]
            print("-------------------------")
            print(f"Starting {signal} pre-processing ({'parallelized' if parallelize[signal] else 'NOT parallelized'})...")
            start_time = time.time()

            if parallelize[signal]:
                with ProcessPoolExecutor(max_workers=12) as executor:
                    futures = {
                        executor.submit(preprocess_file, file_name, input_dir, signal, errors): file_name
                        for file_name in os.listdir(os.path.abspath(input_dir))
                    }

                    for future in as_completed(futures):
                        try:
                            future.result()  
                        except Exception as e:
                            print(f"Error during pre-processing of file: {e}")
            else:
                for file_name in os.listdir(os.path.abspath(input_dir)):
                    preprocess_file(file_name, input_dir, signal, errors)

            end_time = time.time()
            print(f"...{signal} pre-processing completed in {round(end_time - start_time)} s")
            print("-------------------------")

    except KeyboardInterrupt:
        print("User interruption. Terminating ongoing processes...")
        executor.shutdown(wait=False, cancel_futures=True)  
        print("Processes terminated. Exiting.")
        with open("errors.json", "w") as file:
            json.dump({k: list(v) for k, v in errors.items()}, file, indent=4, ensure_ascii=False)
        
        

    with open("errors.json", "w") as file:
        json.dump({k: list(v) for k, v in errors.items()}, file, indent=4, ensure_ascii=False)

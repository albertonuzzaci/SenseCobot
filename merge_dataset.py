from setup import setup, getConfigData
import pandas as pd
import numpy as np
import os

def generate_labels_csv(input_path, output_path): 
	'''
	Reads a CSV file, calculates the averages between TASK_X_A and TASK_X_B for each row,
	and generates a new CSV file with the required columns: Participant_ID, Task_Number, Average.
	'''
	try:
		df = pd.read_csv(input_path)

		output_data = []

		for participant_id in range(1, 22):
			participant_id_str = f"P_{participant_id:02d}"
			for task_number in range(1, 6):
				column_a = f"TASK_{task_number}_A"
				column_b = f"TASK_{task_number}_B"

				if column_a in df.columns and column_b in df.columns:
					average = df.loc[df['Participants'] == participant_id_str, [column_a, column_b]].mean(axis=1).values

					if len(average) > 0:
						output_data.append([participant_id, task_number, average[0]])
				else:
					print(f"Missing columns for task {task_number}: {column_a}, {column_b}")

		df_output = pd.DataFrame(output_data, columns=["Participant_ID", "Task", "Label"])

		df_output.to_csv(output_path, index=False)

	except Exception as e:
		print(f"Error during processing: {e}")


def generate_fused_dataset(input_dir, output_path):
	'''
	Generate a fused dataset from the input directory for all the participants.
	'''
	try:
		all_data = []

		for file_name in os.listdir(input_dir):
			if file_name.endswith('.csv'):
				file_path = os.path.join(input_dir, file_name)
				df = pd.read_csv(file_path)

				# Exclude the "Window Start" column
				if 'Window Start' in df.columns:
					df = df.drop(columns=['Window Start'])

				if 'Timestamp' in df.columns:
					df = df.drop(columns=['Timestamp'])

				# Calculate the mean for each column
				mean_values = df.mean(axis=0).to_dict()

				if "ECG" in file_name:
					participant = int(file_name.split("_")[-3])
					task = 0 if "Baseline" in file_name else int(file_name.split("_")[-5])
     
				if "EEG" in file_name:
					participant = int(file_name.split("_")[-1].split(".")[0])
					task = 0 if "Baseline" in file_name else int(file_name.split("_")[-3])
				if "GSR" in file_name:
					participant = int(file_name.split("_")[-1].split(".")[0])
					task = 0 if "Baseline" in file_name else int(file_name.split("_")[-3][-1])

				if "Emotions" in file_name:
					participant = int(file_name.split("_")[-2])
					task = 0 if "Baseline" in file_name else int(file_name.split("_")[-4])
	 
				# Ensure 'Participant' and 'Task' are the first columns
				mean_values = {'Participant': participant, 'Task': task, **mean_values}

				all_data.append(mean_values)

		# Create a DataFrame from the collected data
		df_output = pd.DataFrame(all_data)

		# Sort by Participant and Task
		df_output = df_output.sort_values(by=['Participant', 'Task'])

		# Save the concatenated DataFrame to the output path
		df_output.to_csv(output_path, index=False)

	except Exception as e:
		print(f"Error during processing: {e}")

if __name__ == "__main__":
	config_data = getConfigData()
	setup(config_data)

	generate_labels_csv(input_path=f"{config_data['LABELS_IN']}",
						output_path=f"{config_data['LABELS_OUT']}")
 
	generate_fused_dataset(input_dir=f"{config_data['PFOLDER']['ECG_PROCESSED_WINDOWS']}",
						output_path=f"{config_data['FINAL_DATASET']['ECG_FINAL_DATASET']}")
 	
	generate_fused_dataset(input_dir=f"{config_data['PFOLDER']['EEG_CSV']}",
						   output_path=f"{config_data['FINAL_DATASET']['EEG_FINAL_DATASET']}")
 
	generate_fused_dataset(input_dir=f"{config_data['PFOLDER']['GSR_PROCESSED_WINDOWS']}",
						output_path=f"{config_data['FINAL_DATASET']['GSR_FINAL_DATASET']}")
 
	generate_fused_dataset(input_dir=f"{config_data['PFOLDER']['EMOTIONS_PROCESSED']}",
						output_path=f"{config_data['FINAL_DATASET']['EMOTIONS_FINAL_DATASET']}")
 
	
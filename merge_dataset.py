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

def generate_final_dataset(output_path):
	'''
	Generate the final dataset by merging the ECG, EEG, GSR, and Emotions datasets.
	'''
	# Load the datasets
	# Check if the datasets exist before reading
	ecg_path = f"{config_data['FINAL_DATASET']['ECG_FINAL_DATASET']}"
	eeg_path = f"{config_data['FINAL_DATASET']['EEG_FINAL_DATASET']}"
	gsr_path = f"{config_data['FINAL_DATASET']['GSR_FINAL_DATASET']}"
	emotions_path = f"{config_data['FINAL_DATASET']['EMOTIONS_FINAL_DATASET']}"

	dataframes = []
	if os.path.exists(ecg_path):
		ecg_df = pd.read_csv(ecg_path)
		dataframes.append(ecg_df)
	if os.path.exists(eeg_path):
		eeg_df = pd.read_csv(eeg_path)
		dataframes.append(eeg_df)
	if os.path.exists(gsr_path):
		gsr_df = pd.read_csv(gsr_path)
		dataframes.append(gsr_df)
	if os.path.exists(emotions_path):
		emotions_df = pd.read_csv(emotions_path)
		dataframes.append(emotions_df)

	labels_df = pd.read_csv(f"{config_data['LABELS_OUT']}")

	# Merge the datasets
	merged_df = dataframes[0]
	for df in dataframes[1:]:
		merged_df = pd.merge(merged_df, df, on=['Participant', 'Task'])
	merged_df = pd.merge(merged_df, labels_df, left_on=['Participant', 'Task'], right_on=['Participant_ID', 'Task'], how='left')
	merged_df['Label'] = merged_df['Label'].fillna('')
 	
	merged_df['Task'] = merged_df['Task'].astype(int)

	
	
	merged_df = merged_df.sort_values(by=['Task', 'Participant'], key=lambda x: (x == 0).astype(int))
	merged_df = merged_df.drop(columns=['Participant_ID'])
 
	# Add "P_" in front of all values in the Participant column
	merged_df['Participant'] = merged_df['Participant'].apply(lambda x: f"P_{x:02d}")
	# Add "Task_" in front of all values in the Task column, replace 0 with "Baseline"
	merged_df['Task'] = merged_df['Task'].apply(lambda x: "Baseline" if x == 0 else f"Task_{x}")
	
	# Round all float values to 4 decimal places
	float_columns = merged_df.select_dtypes(include=['float64']).columns
	merged_df[float_columns] = merged_df[float_columns].round(4)
	# Save the merged dataset
	merged_df.to_csv(output_path, index=False)

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
 
	generate_final_dataset(output_path=f"{config_data['FINAL_DATASET']['FINAL_DATASET_CSV']}")
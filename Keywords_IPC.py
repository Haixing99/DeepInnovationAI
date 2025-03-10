import os
from dataset_process import AIfilter
import config
# Define input and output paths
input_folder = r'./input'
output_folder = r'./Keyword_IPC_filtered'
# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)
# Iterate through all files in input_folder
for filename in os.listdir(input_folder):
    # Check if file is in CSV format
    if filename.endswith('.csv'):
        # Get complete file path
        input_file = os.path.join(input_folder, filename)
        output_file = os.path.join(output_folder, f'results_{filename}')
        aifilter = AIfilter(
            file_path=input_file,                 # File path
            ipc="ipc",                            # IPC column name
            title="title",                        # Title column name
            abstract="abs",                       # Abstract column name
            ipc_list=config.IPC_num_list,         # IPC filter list
            keywords_list=config.keywords_list    # Keywords list
        )
        
        # Load and process data using chunksize parameter, and save results
        aifilter.load_data_in_chunks(chunksize=50000, output_file=output_file)
        print(f'Processed {filename} and saved results to {output_file}')
print("All files have been processed.")
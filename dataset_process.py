import pandas as pd
from pathlib import Path
from tqdm import tqdm
import re

class AIfilter:
    def __init__(self, file_path, ipc, title, abstract, ipc_list, keywords_list, **kwargs):
        # File path, column name for IPC, column name for title, column name for abstract
        self.file_path = file_path
        self.ipc = ipc
        self.title = title
        self.abstract = abstract

        # IPC numbers and Keywords for filtering data
        self.IPC = ipc_list
        self.Keywords = keywords_list
        # This is based on the classification of strategic emerging industries and international patent classification
        self.include_with_exclusions = {
            "A61B5": ["A61B5/0476", "A61B5/0478"]
        }

        # Placeholder for the dataframe
        self.df = None

    def load_data(self):
        # Support for CSV or Excel input files
        # Check file format and load data
        file_ext = Path(self.file_path).suffix
        if file_ext == '.xlsx':
            self.df = pd.read_excel(self.file_path, engine="openpyxl")
        elif file_ext == '.csv':
            self.df = pd.read_csv(self.file_path)
        else:
            raise ValueError("Unsupported file format. Please provide a .csv or .xlsx file.")

        # Handle null values for IPC, Title, and Abstract columns
        self.df[self.ipc] = self.df[self.ipc].fillna('no IPC').astype(str)
        self.df[self.title] = self.df[self.title].fillna('no Title').astype(str)
        self.df[self.abstract] = self.df[self.abstract].fillna('no Abstract').astype(str)

        # Data processing: text is the combination of title and abstract
        self.df['text'] = self.df[self.abstract]
        # Create a new column with all values set to 0, indicating that all patents are initially classified as non-AI
        self.df['predict'] = 0

        # Create a column to explain the reason for classification as 1
        # Reason 1: keywords
        # Reason 2: IPC Number
        self.df['reason'] = "0"

    def ipc_filter(self, ipc_numbers):
        cleaned_ipc_numbers = ipc_numbers.strip("[]").replace('"', '')

        # Convert comma or semicolon separated string to list
        ipc_list = [ipc.strip() for ipc in re.split(r'[;,]', cleaned_ipc_numbers)]

        # Check for exact match and prefix match
        for ipc_number in ipc_list:
            if ipc_number in self.IPC:
                return True, f"Matched IPC: {ipc_number}"
            if ipc_number.startswith('G06F40'):
                return True, f"Matched IPC Prefix: {ipc_number}"

            # Prefix + exclusions
            for prefix, exclusions in self.include_with_exclusions.items():
                if ipc_number.startswith(prefix):
                    if any(ipc_number.startswith(exclusion) for exclusion in exclusions):
                        return False, None
                    return True, f"Matched IPC Prefix with Exclusions: {ipc_number}"
        return False, None

    def keywords_filter(self, text):
        keywords_pattern = r'\b(' + '|'.join([re.escape(keyword) for keyword in self.Keywords]) + r')\b'
        match = re.search(keywords_pattern, text, re.IGNORECASE)
        if match:
            return True, f"Matched Keyword: {match.group(0)}"
        return False, None

    def filter_data(self):
        # Filter using keywords and IPC numbers
        print("########################################")
        print("Filtering using IPC numbers and Keywords")
        print("########################################\n")
        tqdm.pandas(desc="Filtering")
        
        def apply_filters(row):
            ipc_match, ipc_reason = self.ipc_filter(row[self.ipc])
            if ipc_match:
                row['predict'] = 1
                row['reason'] = ipc_reason
                return row

            keyword_match, keyword_reason = self.keywords_filter(row['text'])
            if keyword_match:
                row['predict'] = 1
                row['reason'] = keyword_reason
                return row

            return row

        self.df = self.df.progress_apply(apply_filters, axis=1)

    def load_data_in_chunks(self, chunksize=10000, output_file='result.csv'):
        file_ext = Path(self.file_path).suffix
        data_chunks = pd.read_csv(self.file_path, chunksize=chunksize) if file_ext == '.csv' else pd.read_excel(
            self.file_path, chunksize=chunksize)

        # First create a file to save the final results
        with open(output_file, 'w') as f:
            pass

        chunk_num = 0
        for chunk in data_chunks:
            print(f'Processing Chunk: {chunk_num}')
            chunk[self.ipc] = chunk[self.ipc].fillna('no IPC').astype(str)
            chunk[self.title] = chunk[self.title].fillna('no Title').astype(str)
            chunk[self.abstract] = chunk[self.abstract].fillna('no Abstract').astype(str)

            # Combine title and abstract
            chunk['text'] = 'Title: ' + chunk[self.title] + " " + "Abstract: " + chunk[self.abstract]
            chunk['predict'] = 0
            chunk['reason'] = "0"
            
            print("########################################")
            print(f"Filtering Chunk {chunk_num}")
            print("########################################\n")
            tqdm.pandas(desc="Filtering")

            # Define apply_filters function to process each row in the data chunk
            def apply_filters(row):
                ipc_match, ipc_reason = self.ipc_filter(row[self.ipc])
                if ipc_match:
                    row['predict'] = 1
                    row['reason'] = ipc_reason
                    return row

                keyword_match, keyword_reason = self.keywords_filter(row['text'])
                if keyword_match:
                    row['predict'] = 1
                    row['reason'] = keyword_reason
                    return row

                row['predict'] = 0  # Case of no match
                row['reason'] = "No match"
                return row

            # Apply filter function to current data chunk
            chunk = chunk.progress_apply(apply_filters, axis=1)

            # Only save records that meet the criteria to avoid writing unnecessary data
            filtered_chunk = chunk[chunk['predict'] == 1]

            # Append the filtered chunk to the CSV file
            if not filtered_chunk.empty:
                # mode = 'a' to append to the file
                # header=chunk_num == 0 to only write headers when processing the first chunk
                filtered_chunk.to_csv(output_file, mode='a', index=False, header=chunk_num == 0)

            chunk_num += 1

        print(f"Filtered data has been saved as {output_file}")

    def save_filtered_data(self, output_file='predict.csv'):
        # Save the file. The final prediction results can be viewed in the last column of the saved file
        self.df.to_csv(output_file, index=False)
        print(f"File has been saved as {output_file}")
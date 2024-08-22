import json
import os
import subprocess
import re
from retrievers.main import main  # Import the main function from main.py

def run_command(command):
    """Executes a shell command and returns the cleaned output."""
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command '{' '.join(command)}' failed with return code {result.returncode}")
    return result.stdout.strip()

def extract_path_from_output(output):
    """Extracts the path from the command output, removing any prefix."""
    # Define a regular expression to capture the path after the prefix
    match = re.search(r'(?:Files have been saved to\s)?(.+)', output)
    if match:
        return match.group(1)
    raise ValueError("Unable to extract valid path from output.")

def upload_files(folder_path):
    """Uploads files using the upload.py script and returns the destination folder path."""
    command = ['python3', 'upload.py', folder_path]
    output = run_command(command)
    return extract_path_from_output(output)

def process_documents(destination_folder):
    """Processes documents using the process.py script and returns the path to the JSON output."""
    command = ['python3', 'process.py', destination_folder]
    output = run_command(command)
    return extract_path_from_output(output)

def load_documents(json_output_path):
    """Loads documents from the specified JSON file and ensures proper format."""
    if not os.path.exists(json_output_path):
        raise FileNotFoundError(f"No extracted data file found at {json_output_path}")

    try:
        with open(json_output_path, 'r') as file:
            data = json.load(file)
            print(f"Loaded data: {data}")  # Debugging line to inspect loaded data
            if isinstance(data, dict) and 'error' in data:
                raise RuntimeError(f"Error from process.py: {data['error']}")
            if not isinstance(data, list):
                raise ValueError("JSON data should be a list of documents.")
            for item in data:
                if not isinstance(item, dict):
                    raise ValueError("Each document in the JSON data should be a dictionary.")
            return data
    except json.JSONDecodeError as e:
        raise ValueError(f"Error parsing JSON output: {e}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error loading JSON data: {e}")

def execute_retrieval(documents, query, method, k):
    """Calls the main function from main.py with the provided parameters."""
    return main(documents, query, method, k)

def run_upload_script(folder_path):
    """Handles the upload and processing of documents, and executes retrieval."""
    try:
        # Upload files and get destination folder
        destination_folder = upload_files(folder_path)
        print(f"Destination Folder: {destination_folder}")

        # Process documents and get JSON output path
        json_output_path = process_documents(destination_folder)
        print(f"JSON data has been saved to: {json_output_path}")

        # Load documents from the JSON file
        documents = load_documents(json_output_path)
        print("Extracted Documents:")
        print(json.dumps(documents, indent=2))

        # Define parameters for the retriever call
        query = "Musculoskeletal injury cure"  # Adjust as needed
        method = "bm25"  # Replace with "dpr", "encoder", or any valid Golden Retriever method
        k = 5  # Number of results to retrieve

        # Execute the main function with the retrieved documents
        similar_documents = execute_retrieval(documents, query, method, k)
        for each in similar_documents[0]:
            print(each)
            print(documents[each['id']])

    except RuntimeError as e:
        print(f"Error executing command: {e}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except ValueError as e:
        print(f"Error processing JSON data: {e}")
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON output: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    # Path to the folder with files to upload
    folder_path = '/home/alok/Downloads/sample'
    run_upload_script(folder_path)

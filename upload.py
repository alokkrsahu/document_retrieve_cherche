import os
import argparse
from datetime import datetime

# Function to save uploaded files to a timestamped folder inside "all_files"
def save_files_to_timestamped_folder(folder_path):
    # Create the main "all_files" folder if it doesn't exist
    main_folder = "all_files"
    if not os.path.exists(main_folder):
        os.makedirs(main_folder)

    # Create a timestamped subfolder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    destination_folder = os.path.join(main_folder, timestamp)
    os.makedirs(destination_folder)

    # Walk through the provided folder path and save each file
    for root, _, files in os.walk(folder_path):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            relative_path = os.path.relpath(file_path, folder_path)
            destination_path = os.path.join(destination_folder, relative_path)

            # Create necessary subfolders in the destination path
            os.makedirs(os.path.dirname(destination_path), exist_ok=True)

            # Copy file to the destination path
            with open(file_path, 'rb') as fsrc, open(destination_path, 'wb') as fdst:
                fdst.write(fsrc.read())

    return destination_folder

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Upload files from a folder to a timestamped folder.")
    parser.add_argument('folder_path', type=str, help="Path to the folder containing files to be uploaded")
    args = parser.parse_args()

    # Check if the provided path is a directory
    if not os.path.isdir(args.folder_path):
        print(f"Error: The provided path '{args.folder_path}' is not a valid directory.")
        return

    # Save files to the timestamped folder and get the destination folder path
    destination_folder = save_files_to_timestamped_folder(args.folder_path)

    # Print the path where files were saved
    print(f"Files have been saved to {destination_folder}")

    return destination_folder

if __name__ == "__main__":
    main()

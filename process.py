import os
import json
import argparse
import pdfplumber
from docx import Document
from odf.opendocument import load
from odf.text import P

def extract_paragraphs_from_pdf(file_path):
    paragraphs = []
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    paragraphs.extend(text.split('\n\n'))  # Assuming paragraphs are separated by double newlines
    except Exception as e:
        print(f"Error reading .pdf file '{file_path}': {e}")
    return paragraphs

def extract_paragraphs_from_docx(file_path):
    paragraphs = []
    try:
        doc = Document(file_path)
        for paragraph in doc.paragraphs:
            paragraphs.append(paragraph.text)
    except Exception as e:
        print(f"Error reading .docx file '{file_path}': {e}")
    return paragraphs

def extract_paragraphs_from_odt(file_path):
    paragraphs = []
    try:
        odt_file = load(file_path)
        paragraphs_elements = odt_file.getElementsByType(P)
        for paragraph in paragraphs_elements:
            paragraphs.append(paragraph.textContent)
    except Exception as e:
        print(f"Error reading .odt file '{file_path}': {e}")
    return paragraphs

def extract_text_from_folder(folder_path):
    output = []
    paragraph_id = 1

    # Traverse the folder and subfolders
    for root, _, files in os.walk(folder_path):
        for file_name in files:
            file_path = os.path.join(root, file_name)

            if file_name.endswith('.pdf'):
                paragraphs = extract_paragraphs_from_pdf(file_path)
            elif file_name.endswith('.docx'):
                paragraphs = extract_paragraphs_from_docx(file_path)
            elif file_name.endswith('.odt'):
                paragraphs = extract_paragraphs_from_odt(file_path)
            else:
                print(f"Skipping unsupported file format: {file_name}")
                continue

            for para in paragraphs:
                if para:  # Ensure that we are not adding empty paragraphs
                    output.append({"id": paragraph_id, "text": para})
                    paragraph_id += 1

    return output

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Extract text from documents in a specified folder.")
    parser.add_argument('folder_path', type=str, help="Path to the folder containing the documents")
    args = parser.parse_args()

    # Check if the provided path is a directory
    if not os.path.isdir(args.folder_path):
        error_message = {"error": f"The provided path '{args.folder_path}' is not a valid directory."}
        error_file_path = os.path.join(args.folder_path, 'sys/temp/error_output.json')
        os.makedirs(os.path.dirname(error_file_path), exist_ok=True)
        with open(error_file_path, 'w') as f:
            json.dump(error_message, f, indent=2)
        print(error_file_path)  # Print path for runner.py to capture
        return

    # Create the output directory inside destination_folder if it does not exist
    output_dir = os.path.join(args.folder_path, 'sys/temp')
    os.makedirs(output_dir, exist_ok=True)

    # Process the folder and extract text
    documents = extract_text_from_folder(args.folder_path)

    # Write the extracted text data to a JSON file
    output_file_path = os.path.join(output_dir, 'extracted_data.json')
    with open(output_file_path, 'w') as f:
        json.dump(documents, f, indent=2)

    # Print the path to the JSON file
    print(output_file_path)

if __name__ == "__main__":
    main()

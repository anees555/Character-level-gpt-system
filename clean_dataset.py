import re
import os
import unicodedata

def clean_dataset(text):
    # remove multiple consecutive new lines
    text = re.sub(r'\n{3,}', '\n\n', text)
    # remove leading whitspace form each line
    lines = text.split('\n')
    lines = [line.strip() for line in lines]
    text = '\n'.join(lines)
    text = unicodedata.normalize('NFC', text)
    text = re.sub(r' {2,}', ' ', text)
    # Remove control characters except newline and tab
    text = ''.join(char for char in text if unicodedata.category(char)[0] != 'C' or char in '\n\t')
    # Fix spacing around common punctuation
    text = re.sub(r' +([।,\.!?])', r'\1', text)  # Remove space before punctuation
    text = re.sub(r'([।,\.!?]) +', r'\1 ', text)  # Normalize space after punctuation
    
    return text

def load_dataset(file_path):
    """load the dataset from the file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = f.read()
        print(f"Dataset loaded, {len(data)} characters")
        return data
    except FileNotFoundError:
        print(f"Error! File {file_path} not found")
        return None
    except Exception as e:
        print(f"Error reading file: {e}")
        return None
    
def saved_cleaned_dataset(data, output_path):
    "save cleaned dataset to file"
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(data)
        print(f"saved cleaned dataset: {len(data)} characters")
    except Exception as e:
        print(f"Error saving file: {e}")

def get_dataset_statistics(data):
    """statistics about the dataset"""
    lines = data.strip().split('\n')
    chars = set(data)
    print("---------Dataset statistics ------")
    print(f"Total characters: {len(data)}")
    print(f"Total lines: {len(lines)}")
    print(f"Unique characters(vocab size): {len(chars)}")

def main():

    # file paths
    input_file = "dataset/data-nepali.txt"
    output_file = "dataset/data-nepali-cleaned"

    # load dataset
    data = load_dataset(input_file)
    if data is None:
        return
    print("\n----- Before Cleaning ----")
    get_dataset_statistics(data)
    
    # clean datasest
    cleaned_data = clean_dataset(data)

    print("\n----- After Cleaning -----")
    get_dataset_statistics(cleaned_data)

    # save cleaned dataset 
    print(f"\n--- saving --- ")
    saved_cleaned_dataset(cleaned_data, output_file)

    ## compression ratio calculation
    compression = (1 - len(cleaned_data)/len(data))/ 100
    print(f"\nCompression ration: {compression:.4f}%")

    print(f"\n Dataset cleaning completed")

if __name__ == "__main__":
    main()
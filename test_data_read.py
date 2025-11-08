import pandas as pd
import os

# 設置檔案路徑
file_path = 'Chapter03/datasets/sms_spam_no_header.csv'

print(f"Current working directory: {os.getcwd()}")
print(f"Attempting to read file: {file_path}")

try:
    # 使用我們在 utils.py 中使用的參數
    df = pd.read_csv(
        file_path, 
        encoding='latin-1', 
        header=None, 
        names=['label', 'text'], 
        on_bad_lines='skip', 
        sep=','
    )
    print("--- Read Successful ---")
    print(f"Total rows loaded: {df.shape[0]}")
    print("First 5 rows (including NaN/empty values):")
    print(df.head())
except FileNotFoundError:
    print(f"Error: File not found at the specified path: {file_path}")
    print("Please ensure you run this script from the main 'HW3' directory.")
except Exception as e:
    print(f"Error during manual read: {e}")
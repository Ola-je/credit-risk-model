import pandas as pd
import os

def load_raw_data(file_name):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(base_dir, 'data', 'raw', file_name)

    df = pd.read_csv(file_path)
    print(f"Loaded data from: {file_path}")
    return df
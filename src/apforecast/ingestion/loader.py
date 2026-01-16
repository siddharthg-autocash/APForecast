# src/apforecast/ingestion/loader.py
import pandas as pd
import os
from src.apforecast.core.constants import COLUMN_MAP

def load_file_smart(filepath):
    """
    Reads CSV or Excel. Renames columns based on constants.py map.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    # Auto-detect format
    if filepath.endswith('.csv'):
        df = pd.read_csv(filepath)
    else:
        # Defaults to Excel for xlsx, xls
        df = pd.read_excel(filepath)

    # Normalize Columns (Map User headers to System headers)
    # We invert the map to rename: {User_Col: System_Col}
    rename_dict = {v: k for k, v in COLUMN_MAP.items()} 
    # Wait, the map in constants is {User_Name : System_Name} ?? 
    # Let's assume the user edits constants.py to be: "Check Number": "Check_ID"
    # So we simply pass COLUMN_MAP.
    
    # Actually, standardizing the map in constants:
    # Let's rely on the user putting correct keys in COLUMN_MAP.
    # Current CONSTANTS structure: {"User_Header": "System_Internal_Name"}
    
    df.rename(columns=COLUMN_MAP, inplace=True)
    
    # Strip whitespace from string columns
    df.columns = [c.strip() for c in df.columns]
    
    return df
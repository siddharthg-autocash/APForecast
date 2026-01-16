# src/apforecast/core/config_loader.py
import pandas as pd
from src.apforecast.core.constants import CONFIG_FILE_PATH

def load_vendor_overrides():
    """
    Reads the Excel config file and returns a dictionary of active overrides.
    Key: Vendor_ID
    Value: Dict of strategy parameters
    """
    try:
        df = pd.read_excel(CONFIG_FILE_PATH)
        # Filter for active rules only
        active_rules = df[df['Active'] == True]
        
        overrides = {}
        for _, row in active_rules.iterrows():
            overrides[row['Vendor_ID']] = {
                'Strategy': row['Strategy'],
                'Param_1': row['Param_1'],
                'Param_2': row['Param_2']
            }
        return overrides
    except FileNotFoundError:
        print("Warning: No config file found. Proceeding with defaults.")
        return {}
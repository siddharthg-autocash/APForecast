# src/apforecast/reporting/dashboard.py
import pandas as pd
from src.apforecast.core.constants import REPORTS_DIR

def generate_report(forecast_df, run_date_str):
    """
    forecast_df has columns: [Check_ID, Vendor, Amount, Probability, Expected_Cash]
    """
    filename = f"{REPORTS_DIR}/forecast_{run_date_str}.xlsx"
    
    writer = pd.ExcelWriter(filename, engine='xlsxwriter')
    
    # Summary Sheet
    total_exposure = forecast_df['Amount'].sum()
    expected_outflow = forecast_df['Expected_Cash'].sum()
    
    summary_data = {
        'Metric': ['Total Open AP', 'Expected Cash Outflow (Today)'],
        'Value': [total_exposure, expected_outflow]
    }
    pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
    
    # Detail Sheet
    forecast_df.sort_values(by='Probability', ascending=False, inplace=True)
    forecast_df.to_excel(writer, sheet_name='Check_Details', index=False)
    
    writer.close()
    print(f"Report saved: {filename}")
# 💸 APForecast Engine
A Machine-Learning Driven Cash Forecasting System for Accounts Payable.

This system eliminates the "Cash Blind Spot" by predicting exactly when checks will clear the bank. Instead of relying on static "Due Dates," it uses Bayesian Survival Analysis to calculate the daily clearing probability for every open check based on historical vendor behavior.

# 📂 Project Structure


APForecast/
│
├── data/
│   ├── raw/
│   │   ├── history/            # Place your historical "Cleared Checks.xlsx" here
│   │   └── DD-MM-YYYY/         # Daily folder created automatically (e.g., 16-01-2026)
│   ├── processed/
│   │   └── master_ledger.parquet # The "Brain" (Long-term memory of all transactions)
│   └── config/
│       └── vendor_strategy_overrides.xlsx # The "Rule Book" (User Overrides)
│
├── reports/
│   ├── forecast_DD-MM-YYYY.xlsx  # The Daily Excel Report
│   └── plots/                    # The Visual graphs (Vendor Behavior)
│
├── src/
│   └── apforecast/               # Source Code (Engine, Logic, Math)
│
├── app.py                        # The Streamlit Dashboard (Front-End)
├── create_config.py              # Script to generate the Overrides Template
└── requirements.txt              # Python dependencies

# 🚀 Getting Started
1. Installation
Ensure you have Python 3.10+ installed.

Bash

## Install required libraries
pip install pandas numpy pyarrow openpyxl xlsxwriter seaborn matplotlib streamlit
2. One-Time Setup

A. Initialize the "Brain" (History)

Place your historical cleared checks file in data/raw/history/.

Supported formats: .xlsx or .csv.

Important: Ensure headers match the mapping in src/apforecast/core/constants.py.

B. Generate the "Rule Book" (Overrides) Run this script to create the configuration file for manual rules:

Bash
`
python create_config.py
`

This creates data/config/vendor_strategy_overrides.xlsx.

Edit this Excel file to set fixed rules (e.g., "Always clears on Fridays").

# 🖥️ How to Run (The Workflow)
Option A: The Dashboard (Recommended)
The easiest way to run the system is via the visual interface.

Bash
`
streamlit run app.py
`
Tab 1: History Setup: Upload your history file once to train the model.

Tab 2: Daily Forecast:

Select Today's Date.

Upload your Outstanding Checks file.

(Optional) Upload yesterday's Cleared Checks file to update the brain.

Click RUN FORECAST.

Tab 3: Vendor Intelligence: Inspect specific vendor behavior graphs.

Option B: The Command Line (Headless)
For automation or servers.

Prepare Folders:

Create data/raw/16-01-2026/.

Put Outstanding Checks.xlsx inside.

Run:

Bash
`
python -m src.apforecast.main --date 16-01-2026
`

# 🧠 How It Works (The Logic)
The engine decides the probability of a check clearing using a 3-Step Waterfall:

User Override (The Law):

Checks vendor_strategy_overrides.xlsx.

Example: "Vendor X is set to FIXED_LAG of 7 days." -> Probability = 100% on Day 7.

Specific History (The Specialist):

If the vendor has >5 historical transactions, the system builds a unique probability curve.

Example: "Cintas typically clears in 4-6 days."

Global Cohort (The Safety Net):

If the vendor is new, they are assigned a profile based on check size.

Small (<$10k): Fast clearing profile.

Medium ($10k-$50k): Volatile profile.

Large (>$50k): Slow clearing profile ("The Lazy Giant").

# 📊 The Output
Cash Requirement: A specific dollar amount needed to fund the account today.

Excel Report: A detailed file (reports/forecast_<date>.xlsx) listing every open check and its specific probability of clearing.

Visuals: Reference graphs showing the historical behavior of your vendors.

# 🔧 Configuration (Column Mapping)
To match your specific file headers, edit src/apforecast/core/constants.py:

Python

COLUMN_MAP = {
    "Your File Header Name" : "Check_ID",
    "Another Header"        : "Vendor_ID",
    "Payment Amount"        : "Amount",
    ...
}
Note: The Right Side (System Names) must not change. Only change the Left Side to match your Excel files.
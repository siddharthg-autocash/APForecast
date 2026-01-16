# 💸 APForecast Engine  
### A Machine-Learning Driven Cash Forecasting System for Accounts Payable

APForecast eliminates the **“Cash Blind Spot”** in Accounts Payable by predicting **exactly when checks will clear the bank**.

Instead of relying on static due dates, the system uses **probabilistic modeling (Bayesian-style survival logic)** to compute a **daily clearing probability for every open check**, based on real historical vendor behavior.

This enables finance teams to know **how much cash is actually required today**, not just what is theoretically due.

---

## ✨ Key Features

- 📈 Vendor-specific clearing probability curves  
- 🧠 Persistent historical ledger (“Brain”) across days  
- 📊 Daily Excel forecast reports  
- 📉 Vendor behavior visualizations  
- ⚙️ Manual override rules for special vendors  
- 🖥️ Streamlit dashboard + CLI mode  
- 🧩 Handles new vendors via intelligent cohorting  

---

## 📂 Project Structure

```
APForecast/
│
├── data/
│   ├── raw/
│   │   ├── history/                # Historical cleared checks (one-time setup)
│   │   └── DD-MM-YYYY/             # Daily run folder (auto-created)
│   │       └── Outstanding Checks.xlsx
│   │
│   ├── processed/
│   │   └── master_ledger.parquet   # Long-term memory ("Brain")
│   │
│   └── config/
│       └── vendor_strategy_overrides.xlsx  # Manual rules ("Rule Book")
│
├── reports/
│   ├── forecast_DD-MM-YYYY.xlsx    # Daily forecast output
│   └── plots/                      # Vendor behavior graphs
│
├── src/
│   └── apforecast/
│       ├── core/                   # Dates, constants, utilities
│       ├── ingestion/              # File loaders & cleaners
│       ├── engine/                 # Forecast logic
│       ├── models/                 # Probability & survival models
│       └── main.py                 # CLI entry point
│
├── app.py                          # Streamlit dashboard
├── create_config.py                # Generates override template
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### 1️⃣ Installation

Ensure **Python 3.10+** is installed.

```bash
pip install -r requirements.txt
```

If installing manually:

```bash
pip install pandas numpy pyarrow openpyxl xlsxwriter seaborn matplotlib streamlit
```

---

## 🧠 One-Time Setup

### A. Initialize the History (“Brain”)

1. Place your historical cleared checks file in:

```
data/raw/history/
```

2. Supported formats:
- `.xlsx`
- `.csv`

3. Ensure column headers match the mapping in:

```
src/apforecast/core/constants.py
```

---

### B. Generate the Override Rule Book

Run once:

```bash
python create_config.py
```

This creates:

```
data/config/vendor_strategy_overrides.xlsx
```

Use this file to define **manual vendor rules**, such as:
- Fixed clearing lag
- Specific clearing weekdays

---

## 🖥️ How to Run

### Option A: Streamlit Dashboard (Recommended)

```bash
streamlit run app.py
```

**Tabs**
- History Setup
- Daily Forecast
- Vendor Intelligence

---

### Option B: Command Line (Headless)

```bash
python -m src.apforecast.main --date DD-MM-YYYY
```

---

## 🧠 Forecast Logic (3-Step Waterfall)

1. **User Overrides** – Absolute rules  
2. **Vendor History** – Learned probability curves  
3. **Global Cohorts** – Size-based fallback behavior  

---

## 📊 Outputs

- **Cash Requirement**: Expected cash needed today  
- **Excel Report**: Line-by-line clearing probabilities  
- **Visuals**: Vendor clearing behavior plots  

---

## 🔧 Column Mapping

Edit:
```
src/apforecast/core/constants.py
```

```python
COLUMN_MAP = {
    "Your Excel Header": "System_Field_Name"
}
```

⚠️ Do not change system field names (right-hand side).

---

# # FILE: ./app.py
# import streamlit as st
# import pandas as pd
# import numpy as np
# import os
# import io
# import sys
# import plotly.graph_objects as go
# from datetime import datetime, timedelta

# # --- PATH SETUP ---
# current_dir = os.path.dirname(os.path.abspath(__file__))
# if current_dir not in sys.path:
#     sys.path.append(current_dir)

# # --- IMPORTS ---
# try:
#     from src.apforecast.core.constants import *
#     from src.apforecast.ingestion.reconciler import ingest_and_reconcile
#     from src.apforecast.modeling.engine import ForecastEngine
#     from src.apforecast.core.config_loader import load_vendor_overrides
#     from experiments.backtesting.core import run_walk_forward_backtest, plot_backtest_results
# except ImportError as e:
#     st.error(f"❌ Import Error: {e}")
#     st.stop()

# # --- PAGE CONFIG ---
# st.set_page_config(page_title="APForecast Commander", layout="wide", page_icon="💸")
# st.title("💸 APForecast Commander")

# # --- SESSION STATE ---
# if 'ledger' not in st.session_state: st.session_state['ledger'] = None
# if 'forecast_df' not in st.session_state: st.session_state['forecast_df'] = None
# if 'bt_clean' not in st.session_state: st.session_state['bt_clean'] = None
# if 'bt_dates' not in st.session_state: st.session_state['bt_dates'] = None
# if 'bt_res' not in st.session_state: st.session_state['bt_res'] = None

# # ==========================================
# # 0. DATA UTILITIES (The "Column Hunter")
# # ==========================================
# def smart_normalize_columns(df):
#     """
#     Robustly finds columns.
#     PRIORITY: Forces 'Reference' (Column G) to be Vendor if present.
#     """
#     # 1. Clean headers: strip whitespace and convert to string
#     df.columns = [str(c).strip() for c in df.columns]
    
#     rename_map = {}
#     found_vendor = False
    
#     # --- PRE-SCAN: The "Reference" Force ---
#     # We check for 'Reference' specifically before the main loop to ensure 
#     # generic keywords like 'Name' in other columns don't steal the spot.
#     for col in df.columns:
#         c_lower = col.lower()
#         if 'reference' in c_lower and 'bacs' not in c_lower:
#             rename_map[col] = COL_VENDOR_ID
#             found_vendor = True
#             break 

#     # 2. Main Search Logic 
#     for col in df.columns:
#         if col in rename_map: continue
#         c_lower = col.lower()
        
#         # Vendor Fallback (only if Reference wasn't found)
#         if not found_vendor:
#             if any(x in c_lower for x in ['vendor', 'payee', 'beneficiary', 'name', 'description']):
#                 rename_map[col] = COL_VENDOR_ID
#                 found_vendor = True
#                 continue

#         # Other Columns
#         if any(x in c_lower for x in ['amount', 'debit', 'payment']):
#             if COL_AMOUNT not in rename_map.values():
#                 rename_map[col] = COL_AMOUNT
#                 continue
#         if any(x in c_lower for x in ['check', 'num', 'doc']):
#             if 'amount' not in c_lower and COL_CHECK_ID not in rename_map.values():
#                 rename_map[col] = COL_CHECK_ID
#                 continue
#         if 'clear' in c_lower and 'date' in c_lower:
#             rename_map[col] = 'Clear_Date'
#         elif 'post' in c_lower or 'txn' in c_lower or c_lower == 'date':
#             rename_map[col] = COL_POST_DATE

#     # 3. Apply Renames
#     df = df.rename(columns=rename_map)
    
#     # --- FIX 1: DATE PARSING (Day First) ---
#     # We explicitly tell pandas to try 'dayfirst=True' for formats like 22-07-2025
#     date_cols = [c for c in df.columns if c in [COL_POST_DATE, 'Clear_Date']]
#     for dc in date_cols:
#         df[dc] = pd.to_datetime(df[dc], dayfirst=True, errors='coerce')

#     # --- FIX 2: VENDOR NORMALIZATION ---
#     # Consolidate "American Hose -" and "AMERICAN HOSE" into one ID
#     if COL_VENDOR_ID in df.columns:
#         df[COL_VENDOR_ID] = df[COL_VENDOR_ID].astype(str).str.upper().str.strip(" -_.")

#     # 4. Final Safety Check
#     if COL_VENDOR_ID not in df.columns:
#         df[COL_VENDOR_ID] = "Unknown_Vendor"
#         if not df.empty:
#             st.warning(f"⚠️ Column Warning: Could not find Vendor/Reference column.")
    
#     return df

# # ==========================================
# # 1. VISUALIZATION HELPERS
# # ==========================================

# def plot_vendor_history_legacy(model, vendor_name, ledger):
#     """ Classic History Plot """
#     history = ledger[(ledger[COL_VENDOR_ID] == vendor_name) & (ledger['Status'] == 'CLEARED')].copy()
#     if history.empty: return None

#     history['Days_Taken'] = (history['Clear_Date'] - history[COL_POST_DATE]).dt.days
#     history = history[history['Days_Taken'] >= 0]
#     if history.empty: return None

#     daily_stats = history.groupby('Days_Taken').agg({
#         COL_AMOUNT: 'sum',
#         COL_CHECK_ID: list,
#         COL_VENDOR_ID: 'count'
#     }).rename(columns={COL_VENDOR_ID: 'Count'}).reset_index()

#     daily_stats['Probability'] = daily_stats['Count'] / daily_stats['Count'].sum()
#     daily_stats['Label'] = daily_stats[COL_AMOUNT].apply(lambda x: f"${x/1000:.0f}k" if x >= 1000 else f"${int(x)}")

#     def build_hover(row):
#         ids = sorted([str(x) for x in row[COL_CHECK_ID]])[:5]
#         id_str = "<br>• " + "<br>• ".join(ids)
#         return (f"<b>Day {row['Days_Taken']}</b><br>Prob: {row['Probability']:.1%}<br>"
#                 f"Vol: <b>${row[COL_AMOUNT]:,.0f}</b><br>IDs:{id_str}")
#     daily_stats['Hover_Text'] = daily_stats.apply(build_hover, axis=1)

#     fig = go.Figure()
#     fig.add_trace(go.Bar(
#         x=daily_stats['Days_Taken'], y=daily_stats['Probability'],
#         text=daily_stats['Label'], textposition='outside',
#         marker_color='#E67E22', opacity=0.6,
#         hovertext=daily_stats['Hover_Text'], hoverinfo="text",
#         name="Hist. Probability"
#     ))
    
#     x_range = np.arange(0, daily_stats['Days_Taken'].max() + 5)
#     fig.add_trace(go.Scatter(
#         x=x_range, y=[model.cdf(x) for x in x_range],
#         name="Cumulative (CDF)", mode='lines',
#         line=dict(color='#154360', width=3), yaxis="y2"
#     ))

#     fig.update_layout(
#         title=f"<b>Payment Profile: {vendor_name}</b>",
#         yaxis=dict(title="Probability", tickformat=".0%"),
#         yaxis2=dict(title="Cumulative %", overlaying="y", side="right", showgrid=False),
#         template="plotly_white", height=450, showlegend=False
#     )
#     return fig

# def plot_snowball_interactive(engine, open_checks, run_date, custom_delay=0):
#     """ Forecast Plot with SHADOW IDENTITY FALLBACK """
#     fig = go.Figure()
#     dates = [run_date + timedelta(days=i) for i in range(14)]
    
#     base_cash = []
#     hover_texts = []
    
#     for d in dates:
#         day_total = 0
#         contributors = []
#         for _, row in open_checks.iterrows():
#             # 1. Try Specific Prediction
#             prob = engine.predict_check(row, d, current_date_override=run_date - timedelta(days=1))
#             sim_age = (d - row[COL_POST_DATE]).days
            
#             # 2. Shadow Identity Fallback
#             # If Specific Model returns 0% (because check is older than history),
#             # we pretend the vendor is 'Unknown_Vendor' to force the Global/Generic curve.
#             if prob < 0.001 and sim_age > 0:
#                 shadow_row = row.copy()
#                 shadow_row[COL_VENDOR_ID] = "Unknown_Vendor"
#                 # Ask engine to predict for 'Unknown_Vendor'
#                 prob_global = engine.predict_check(shadow_row, d, current_date_override=run_date - timedelta(days=1))
                
#                 # Only use global if it offers a better probability
#                 if prob_global > prob:
#                     prob = prob_global

#             expected = row[COL_AMOUNT] * prob
            
#             if expected > 0.50: 
#                 day_total += expected
#                 contributors.append((row[COL_CHECK_ID], row[COL_AMOUNT], sim_age, prob))
        
#         base_cash.append(day_total)
        
#         # Tooltip Logic
#         contributors.sort(key=lambda x: x[1], reverse=True)
#         top_list = contributors[:20] 
        
#         hover_html = f"<b>{d.strftime('%b-%d')}</b><br>Expected: <b>${day_total:,.0f}</b><br><br>Checks Contributing:"
#         for cid, amt, age, p in top_list:
#             hover_html += f"<br>• #{cid} (${amt:,.0f}) | Age: {age}d | Prob: {p:.0%}"
#         if len(contributors) > 20:
#             hover_html += f"<br><i>...and {len(contributors)-20} more</i>"
#         hover_texts.append(hover_html)

#     # Base Case Bar
#     fig.add_trace(go.Bar(
#         x=[d.strftime('%b-%d') for d in dates], 
#         y=base_cash,
#         name="Expected Cash Flow",
#         marker_color='#F5B041', opacity=0.8,
#         text=[f"${v/1000:.0f}k" if v > 1000 else "" for v in base_cash],
#         textposition='outside',
#         hovertext=hover_texts, hoverinfo="text"
#     ))

#     # Scenario Line
#     if custom_delay > 0:
#         delay_cash = []
#         for d in dates:
#             sim_date = d - timedelta(days=custom_delay)
#             day_total = 0
#             for _, row in open_checks.iterrows():
#                 if sim_date < run_date: prob = 0
#                 else:
#                     # Scenario: Specific
#                     prob = engine.predict_check(row, sim_date, current_date_override=run_date - timedelta(days=1))
                    
#                     # Scenario: Shadow Fallback
#                     sim_age_scen = (sim_date - row[COL_POST_DATE]).days
#                     if prob < 0.001 and sim_age_scen > 0:
#                         shadow_row = row.copy()
#                         shadow_row[COL_VENDOR_ID] = "Unknown_Vendor"
#                         prob_global = engine.predict_check(shadow_row, sim_date, current_date_override=run_date - timedelta(days=1))
#                         if prob_global > prob:
#                             prob = prob_global
                            
#                 day_total += (row[COL_AMOUNT] * prob)
#             delay_cash.append(day_total)
            
#         fig.add_trace(go.Scatter(
#             x=[d.strftime('%b-%d') for d in dates], 
#             y=delay_cash,
#             name=f"Scenario: Delay +{custom_delay} Days",
#             mode='lines+markers',
#             line=dict(width=3, color='#27AE60'),
#             marker=dict(size=6)
#         ))

#     fig.update_layout(
#         title="<b>Cash Flow Forecast</b>",
#         xaxis_title="Future Date", yaxis_title="Expected Outflow ($)",
#         template="plotly_white", height=500, hovermode="x unified",
#         legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.8)')
#     )
#     return fig

# def plot_interactive_landscape(ledger, run_date):
#     """ Colorful Jitter """
#     open_checks = ledger[ledger[COL_STATUS] == STATUS_OPEN].copy()
#     if open_checks.empty: return None
#     open_checks['Age'] = (run_date - open_checks[COL_POST_DATE]).dt.days
    
#     fig = go.Figure()
#     fig.add_trace(go.Scatter(
#         x=open_checks['Age'], y=open_checks[COL_AMOUNT], mode='markers',
#         marker=dict(
#             size=open_checks[COL_AMOUNT].apply(lambda x: np.log(x)*3 if x>0 else 5),
#             color=open_checks['Age'], colorscale='Portland', showscale=True,
#             line=dict(width=1, color='DarkSlateGrey')
#         ),
#         text=open_checks[COL_VENDOR_ID],
#         hovertemplate="<b>%{text}</b><br>$%{y:,.2f}<br>%{x} days old"
#     ))
#     fig.update_layout(
#         title="<b>Outstanding Check Landscape</b><br><sup>Color = Age | Size = Amount</sup>",
#         xaxis_title="Days Since Posted", yaxis_title="Amount ($)",
#         template="plotly_white", height=450
#     )
#     return fig

# # ==========================================
# # 2. MAIN APP
# # ==========================================
# mode = st.sidebar.radio("Navigation", ["🚀 Forecast & Intelligence", "🧪 Backtest Lab"])

# if mode == "🚀 Forecast & Intelligence":
#     st.header("🚀 Daily Cash Forecast")
    
#     # --- UPDATE LEDGER FEATURE ---
#     with st.expander("🔄 Update Ledger History (Add Cleared Checks)", expanded=False):
#         st.caption("Upload a file with recently cleared checks. The system will remove duplicates and add new ones to history.")
#         update_file = st.file_uploader("Upload Cleared Checks File", type=['xlsx', 'csv'], key="update_upl")
        
#         if update_file and st.button("Merge into History"):
#             with st.spinner("Merging..."):
#                 try:
#                     # 1. Load the new file
#                     new_df = pd.read_excel(update_file) if update_file.name.endswith('.xlsx') else pd.read_csv(update_file)
#                     new_df = smart_normalize_columns(new_df)
                    
#                     # 2. Load the existing Master Ledger (assuming it sits in data/master.xlsx)
#                     # NOTE: Adjust path to where your master file actually lives
#                     master_path = "data/master_ledger.xlsx"
#                     if os.path.exists(master_path):
#                         master_df = pd.read_excel(master_path)
#                     else:
#                         master_df = pd.DataFrame()
                        
#                     if not master_df.empty:
#                         master_df = smart_normalize_columns(master_df)

#                     # 3. Deduplicate
#                     if COL_CHECK_ID in new_df.columns and COL_CHECK_ID in master_df.columns:
#                         existing_ids = set(master_df[COL_CHECK_ID].astype(str))
                        
#                         # Filter for rows where Check ID is NOT in existing
#                         unique_new = new_df[~new_df[COL_CHECK_ID].astype(str).isin(existing_ids)]
                        
#                         if not unique_new.empty:
#                             # Append
#                             updated_master = pd.concat([master_df, unique_new], ignore_index=True)
#                             updated_master.to_excel(master_path, index=False)
#                             st.success(f"✅ Success! Added {len(unique_new)} new cleared checks to history.")
#                         else:
#                             st.info("ℹ️ No new checks found. All checks in file already exist in history.")
#                     else:
#                         # Fallback if starting fresh
#                         new_df.to_excel(master_path, index=False)
#                         st.success(f"✅ Created new Master Ledger with {len(new_df)} rows.")
                        
#                 except Exception as e:
#                     st.error(f"Error updating ledger: {e}")

#     # --- FORECAST INPUTS ---
#     with st.expander("📂 Run Daily Forecast", expanded=True):
#         c1, c2 = st.columns(2)
#         run_date = c1.date_input("Run Date", value="today")
#         run_date_pd = pd.to_datetime(run_date)
#         uncleared_file = c2.file_uploader("Outstanding Checks (Required)", type=['xlsx', 'csv'])
        
#         if c2.button("RUN FORECAST", type="primary"):
#             if uncleared_file:
#                 with st.spinner("Processing..."):
#                     os.makedirs(f"data/raw/{run_date}", exist_ok=True)
#                     with open(f"data/raw/{run_date}/uncleared_checks.xlsx", "wb") as f: f.write(uncleared_file.getbuffer())
                    
#                     ledger = ingest_and_reconcile(str(run_date), run_date_pd)
                    
#                     if COL_VENDOR_ID not in ledger.columns or (ledger[COL_VENDOR_ID] == 'Unknown_Vendor').all():
#                         ledger = smart_normalize_columns(ledger)

#                     st.session_state['ledger'] = ledger
#                     overrides = load_vendor_overrides()
#                     engine = ForecastEngine(ledger, overrides)
                    
#                     open_checks = ledger[ledger[COL_STATUS] == STATUS_OPEN].copy()
#                     data = []
                    
#                     # --- CLEAN 2-STAGE PREDICTION LOOP ---
#                     for _, row in open_checks.iterrows():
#                         # 1. Specific Prediction
#                         prob = engine.predict_check(row, run_date_pd, current_date_override=run_date_pd-timedelta(days=1))
                        
#                         # 2. Shadow Identity Fallback (Stage 2)
#                         # If Specific Model returns 0% (likely older than history),
#                         # force Generic/Global model by pretending to be "Unknown_Vendor"
#                         sim_age = (run_date_pd - row[COL_POST_DATE]).days
#                         if prob < 0.001 and sim_age > 0:
#                             shadow_row = row.copy()
#                             shadow_row[COL_VENDOR_ID] = "Unknown_Vendor"
                            
#                             prob_global = engine.predict_check(shadow_row, run_date_pd, current_date_override=run_date_pd-timedelta(days=1))
                            
#                             if prob_global > prob:
#                                 prob = prob_global

#                         data.append({**row, 'Probability': prob, 'Expected_Cash': row[COL_AMOUNT]*prob})
                    
#                     st.session_state['forecast_df'] = pd.DataFrame(data)
#                     st.success("Done!")

#     if st.session_state['forecast_df'] is not None:
#         df = st.session_state['forecast_df']
#         ledger = st.session_state['ledger']
        
#         st.markdown("### 📊 Portfolio Overview")
#         c1, c2 = st.columns([1, 3])
#         with c1:
#             st.metric("CASH REQUIRED TODAY", f"${df['Expected_Cash'].sum():,.2f}")
#             st.metric("TOTAL EXPOSURE", f"${df[COL_AMOUNT].sum():,.2f}")
#             buffer = io.BytesIO()
#             with pd.ExcelWriter(buffer) as writer: df.to_excel(writer, index=False)
#             st.download_button("📥 Download Excel", buffer.getvalue(), f"forecast_{run_date}.xlsx")
#         with c2:
#             st.plotly_chart(plot_interactive_landscape(ledger, run_date_pd), use_container_width=True)

#         st.markdown("---")
#         st.subheader("🕵️ Vendor Intelligence")
        
#         valid_vendors = sorted([v for v in df[COL_VENDOR_ID].unique() if v != "Unknown_Vendor"])
#         if not valid_vendors: valid_vendors = ["Unknown_Vendor"]
        
#         sel_vendor = st.selectbox("Select Vendor:", valid_vendors)
        
#         overrides = load_vendor_overrides()
#         engine = ForecastEngine(ledger, overrides)
        
#         if sel_vendor:
#             c1, c2 = st.columns(2)
#             with c1:
#                 if sel_vendor in engine.models['SPECIFIC']:
#                     st.plotly_chart(plot_vendor_history_legacy(engine.models['SPECIFIC'][sel_vendor], sel_vendor, ledger), use_container_width=True)
#                 else:
#                     st.info("Insufficient history for this vendor.")
            
#             with c2:
#                 delay_days = st.slider(f"Simulate Delay for {sel_vendor}", 0, 14, 2)
#                 v_open = ledger[(ledger[COL_STATUS]==STATUS_OPEN) & (ledger[COL_VENDOR_ID]==sel_vendor)].copy()
#                 if not v_open.empty:
#                     st.plotly_chart(plot_snowball_interactive(engine, v_open, run_date_pd, custom_delay=delay_days), use_container_width=True)
#                     st.info(f"**Insight:** Moving the slider shows how cash flow shifts if {sel_vendor} is delayed.")
#                     with st.expander("See Raw Data"):
#                         v_open['Age'] = (run_date_pd - v_open[COL_POST_DATE]).dt.days
#                         st.dataframe(v_open[[COL_CHECK_ID, COL_AMOUNT, COL_POST_DATE, 'Age']].sort_values('Age', ascending=False), use_container_width=True)
#                 else:
#                     st.success("No open checks.")

# # ------------------------------------------
# # TAB 2: BACKTEST LAB
# # ------------------------------------------
# elif mode == "🧪 Backtest Lab":
#     st.title("🧪 Backtest Lab")
    
#     with st.expander("⚙️ Configuration", expanded=True):
#         c1, c2 = st.columns(2)
#         start_dt = c1.date_input("Start Date", value=datetime.today()-timedelta(days=90))
#         end_dt = c1.date_input("End Date", value=datetime.today()-timedelta(days=1))
#         hist_file = c2.file_uploader("Upload Master History", type=['xlsx'])

#     if hist_file and st.button("RUN GLOBAL BACKTEST", type="primary"):
#         with st.spinner("Processing History..."):
#             try:
#                 raw = pd.read_excel(hist_file)
#                 cols_str = "".join([str(c) for c in raw.columns])
#                 if "Unnamed" in cols_str or raw.shape[1] < 3:
#                     raw = pd.read_excel(hist_file, header=1)
#             except Exception as e:
#                 st.error(f"Error reading file: {e}")
#                 st.stop()

#             clean = smart_normalize_columns(raw)
#             if COL_POST_DATE in clean.columns:
#                 clean[COL_POST_DATE] = pd.to_datetime(clean[COL_POST_DATE], dayfirst=True, errors='coerce')
#             if 'Clear_Date' in clean.columns:
#                 clean['Clear_Date'] = pd.to_datetime(clean['Clear_Date'], dayfirst=True, errors='coerce')
            
#             # --- CALCULATE DAYS_TO_SETTLE FOR ENGINE TRAINING ---
#             if 'Clear_Date' in clean.columns and COL_POST_DATE in clean.columns:
#                 clean['Days_to_Settle'] = (clean['Clear_Date'] - clean[COL_POST_DATE]).dt.days
#                 # Ensure status is cleared where appropriate
#                 mask_cleared = clean['Clear_Date'].notna()
#                 clean.loc[mask_cleared, 'Status'] = 'CLEARED'
            
#             st.session_state['bt_clean'] = clean
#             st.session_state['bt_dates'] = (start_dt, end_dt)
            
#             overrides = load_vendor_overrides()
#             res = run_walk_forward_backtest(clean, str(start_dt), str(end_dt), overrides)
#             st.session_state['bt_res'] = res
#             st.success("Global Backtest Complete.")

#     if st.session_state['bt_res'] is not None:
#         res = st.session_state['bt_res']
        
#         st.markdown("### 🌍 Global Results")
#         m1, m2, m3 = st.columns(3)
#         act = res['Actual'].sum()
#         pred = res['Predicted'].sum()
#         var = pred - act
#         m1.metric("Actual", f"${act:,.0f}")
#         m2.metric("Predicted", f"${pred:,.0f}")
#         m3.metric("Variance", f"${var:,.0f}", delta=f"{(var/act)*100:.1f}%")
        
#         st.plotly_chart(plot_backtest_results(res, "Global"), use_container_width=True)
        
#         st.markdown("---")
#         st.subheader("🔬 Vendor Drill-Down")
#         clean_df = st.session_state.get('bt_clean')
        
#         if clean_df is not None:
#             vendors = sorted([v for v in clean_df[COL_VENDOR_ID].astype(str).unique() if v != "Unknown_Vendor"])
#             if not vendors: vendors = ["Unknown_Vendor"]

#             c_sel, c_btn = st.columns([3, 1])
#             v_sel = c_sel.selectbox("Select Vendor:", vendors)
            
#             if c_btn.button(f"Run for {v_sel}"):
#                 s, e = st.session_state['bt_dates']
#                 overrides = load_vendor_overrides()
#                 with st.spinner(f"Simulating {v_sel}..."):
#                     v_res = run_walk_forward_backtest(clean_df, str(s), str(e), overrides, vendor_filter=v_sel)
#                     if not v_res.empty:
#                         v_act = v_res['Actual'].sum()
#                         v_pred = v_res['Predicted'].sum()
#                         v_var = v_pred - v_act
#                         c1, c2, c3 = st.columns(3)
#                         c1.metric(f"{v_sel} Actual", f"${v_act:,.0f}")
#                         c2.metric(f"{v_sel} Predicted", f"${v_pred:,.0f}")
#                         c3.metric("Variance", f"${v_var:,.0f}")
#                         st.plotly_chart(plot_backtest_results(v_res, v_sel), use_container_width=True)
#                     else:
#                         st.warning("No cleared checks found for this vendor in this period.")

# ====================
# FILE: ./app.py
# ====================
import streamlit as st
import pandas as pd
import numpy as np
import os
import io
import sys
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# --- IMPORTS ---
try:
    from src.apforecast.core.constants import *
    from src.apforecast.ingestion.reconciler import ingest_and_reconcile
    from src.apforecast.modeling.engine import ForecastEngine
    from src.apforecast.core.config_loader import load_vendor_overrides
    from experiments.backtesting.core import run_walk_forward_backtest, plot_backtest_results
except ImportError as e:
    st.error(f"❌ Import Error: {e}")
    st.stop()

# --- PAGE CONFIG ---
st.set_page_config(page_title="APForecast Commander", layout="wide", page_icon="💸")
st.title("💸 APForecast Commander")

# --- SESSION STATE ---
if 'ledger' not in st.session_state: st.session_state['ledger'] = None
if 'forecast_df' not in st.session_state: st.session_state['forecast_df'] = None
if 'bt_clean' not in st.session_state: st.session_state['bt_clean'] = None
if 'bt_dates' not in st.session_state: st.session_state['bt_dates'] = None
if 'bt_res' not in st.session_state: st.session_state['bt_res'] = None

# ==========================================
# 0. DATA UTILITIES (The "Column Hunter")
# ==========================================
def smart_normalize_columns(df):
    """
    Robustly finds columns.
    PRIORITY: Forces 'Reference' (Column G) to be Vendor if present.
    """
    # 1. Clean headers: strip whitespace and convert to string
    df.columns = [str(c).strip() for c in df.columns]
    
    rename_map = {}
    found_vendor = False
    
    # --- PRE-SCAN: The "Reference" Force ---
    # We check for 'Reference' specifically before the main loop to ensure 
    # generic keywords like 'Name' in other columns don't steal the spot.
    for col in df.columns:
        c_lower = col.lower()
        if 'reference' in c_lower and 'bacs' not in c_lower:
            rename_map[col] = COL_VENDOR_ID
            found_vendor = True
            break 

    # 2. Main Search Logic 
    for col in df.columns:
        if col in rename_map: continue
        c_lower = col.lower()
        
        # Vendor Fallback (only if Reference wasn't found)
        if not found_vendor:
            if any(x in c_lower for x in ['vendor', 'payee', 'beneficiary', 'name', 'description']):
                rename_map[col] = COL_VENDOR_ID
                found_vendor = True
                continue

        # Other Columns
        if any(x in c_lower for x in ['amount', 'debit', 'payment']):
            if COL_AMOUNT not in rename_map.values():
                rename_map[col] = COL_AMOUNT
                continue
        if any(x in c_lower for x in ['check', 'num', 'doc']):
            if 'amount' not in c_lower and COL_CHECK_ID not in rename_map.values():
                rename_map[col] = COL_CHECK_ID
                continue
        if 'clear' in c_lower and 'date' in c_lower:
            rename_map[col] = 'Clear_Date'
        elif 'post' in c_lower or 'txn' in c_lower or c_lower == 'date':
            rename_map[col] = COL_POST_DATE

    # 3. Apply Renames
    df = df.rename(columns=rename_map)
    
    # --- FIX 1: DATE PARSING (Day First) ---
    # We explicitly tell pandas to try 'dayfirst=True' for formats like 22-07-2025
    date_cols = [c for c in df.columns if c in [COL_POST_DATE, 'Clear_Date']]
    for dc in date_cols:
        df[dc] = pd.to_datetime(df[dc], dayfirst=True, errors='coerce')

    # --- FIX 2: VENDOR NORMALIZATION ---
    # Consolidate "American Hose -" and "AMERICAN HOSE" into one ID
    if COL_VENDOR_ID in df.columns:
        df[COL_VENDOR_ID] = df[COL_VENDOR_ID].astype(str).str.upper().str.strip(" -_.")

    # 4. Final Safety Check
    if COL_VENDOR_ID not in df.columns:
        df[COL_VENDOR_ID] = "Unknown_Vendor"
        if not df.empty:
            st.warning(f"⚠️ Column Warning: Could not find Vendor/Reference column.")
    
    return df

# ==========================================
# 1. VISUALIZATION HELPERS
# ==========================================

def plot_vendor_history_legacy(model, vendor_name, ledger):
    """ Classic History Plot """
    history = ledger[(ledger[COL_VENDOR_ID] == vendor_name) & (ledger['Status'] == 'CLEARED')].copy()
    if history.empty: return None

    history['Days_Taken'] = (history['Clear_Date'] - history[COL_POST_DATE]).dt.days
    history = history[history['Days_Taken'] >= 0]
    if history.empty: return None

    daily_stats = history.groupby('Days_Taken').agg({
        COL_AMOUNT: 'sum',
        COL_CHECK_ID: list,
        COL_VENDOR_ID: 'count'
    }).rename(columns={COL_VENDOR_ID: 'Count'}).reset_index()

    daily_stats['Probability'] = daily_stats['Count'] / daily_stats['Count'].sum()
    daily_stats['Label'] = daily_stats[COL_AMOUNT].apply(lambda x: f"${x/1000:.0f}k" if x >= 1000 else f"${int(x)}")

    def build_hover(row):
        ids = sorted([str(x) for x in row[COL_CHECK_ID]])[:5]
        id_str = "<br>• " + "<br>• ".join(ids)
        return (f"<b>Day {row['Days_Taken']}</b><br>Prob: {row['Probability']:.1%}<br>"
                f"Vol: <b>${row[COL_AMOUNT]:,.0f}</b><br>IDs:{id_str}")
    daily_stats['Hover_Text'] = daily_stats.apply(build_hover, axis=1)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=daily_stats['Days_Taken'], y=daily_stats['Probability'],
        text=daily_stats['Label'], textposition='outside',
        marker_color='#E67E22', opacity=0.6,
        hovertext=daily_stats['Hover_Text'], hoverinfo="text",
        name="Hist. Probability"
    ))
    
    x_range = np.arange(0, daily_stats['Days_Taken'].max() + 5)
    fig.add_trace(go.Scatter(
        x=x_range, y=[model.cdf(x) for x in x_range],
        name="Cumulative (CDF)", mode='lines',
        line=dict(color='#154360', width=3), yaxis="y2"
    ))

    fig.update_layout(
        title=f"<b>Payment Profile: {vendor_name}</b>",
        yaxis=dict(title="Probability", tickformat=".0%"),
        yaxis2=dict(title="Cumulative %", overlaying="y", side="right", showgrid=False),
        template="plotly_white", height=450, showlegend=False
    )
    return fig

def plot_snowball_interactive(engine, open_checks, run_date, custom_delay=0):
    """ Forecast Plot """
    fig = go.Figure()
    dates = [run_date + timedelta(days=i) for i in range(14)]
    
    base_cash = []
    hover_texts = []
    
    for d in dates:
        day_total = 0
        contributors = []
        for _, row in open_checks.iterrows():
            # The Engine now handles Specific -> Global fallback internally
            prob = engine.predict_check(row, d, current_date_override=run_date - timedelta(days=1))
            sim_age = (d - row[COL_POST_DATE]).days
            
            expected = row[COL_AMOUNT] * prob
            
            if expected > 0.50: 
                day_total += expected
                contributors.append((row[COL_CHECK_ID], row[COL_AMOUNT], sim_age, prob))
        
        base_cash.append(day_total)
        
        # Tooltip Logic
        contributors.sort(key=lambda x: x[1], reverse=True)
        top_list = contributors[:20] 
        
        hover_html = f"<b>{d.strftime('%b-%d')}</b><br>Expected: <b>${day_total:,.0f}</b><br><br>Checks Contributing:"
        for cid, amt, age, p in top_list:
            hover_html += f"<br>• #{cid} (${amt:,.0f}) | Age: {age}d | Prob: {p:.0%}"
        if len(contributors) > 20:
            hover_html += f"<br><i>...and {len(contributors)-20} more</i>"
        hover_texts.append(hover_html)

    # Base Case Bar
    fig.add_trace(go.Bar(
        x=[d.strftime('%b-%d') for d in dates], 
        y=base_cash,
        name="Expected Cash Flow",
        marker_color='#F5B041', opacity=0.8,
        text=[f"${v/1000:.0f}k" if v > 1000 else "" for v in base_cash],
        textposition='outside',
        hovertext=hover_texts, hoverinfo="text"
    ))

    # Scenario Line
    if custom_delay > 0:
        delay_cash = []
        for d in dates:
            sim_date = d - timedelta(days=custom_delay)
            day_total = 0
            for _, row in open_checks.iterrows():
                if sim_date < run_date: prob = 0
                else:
                    # The Engine handles the fallback here too
                    prob = engine.predict_check(row, sim_date, current_date_override=run_date - timedelta(days=1))
                            
                day_total += (row[COL_AMOUNT] * prob)
            delay_cash.append(day_total)
            
        fig.add_trace(go.Scatter(
            x=[d.strftime('%b-%d') for d in dates], 
            y=delay_cash,
            name=f"Scenario: Delay +{custom_delay} Days",
            mode='lines+markers',
            line=dict(width=3, color='#27AE60'),
            marker=dict(size=6)
        ))

    fig.update_layout(
        title="<b>Cash Flow Forecast</b>",
        xaxis_title="Future Date", yaxis_title="Expected Outflow ($)",
        template="plotly_white", height=500, hovermode="x unified",
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.8)')
    )
    return fig

def plot_interactive_landscape(ledger, run_date):
    """ Colorful Jitter """
    open_checks = ledger[ledger[COL_STATUS] == STATUS_OPEN].copy()
    if open_checks.empty: return None
    open_checks['Age'] = (run_date - open_checks[COL_POST_DATE]).dt.days
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=open_checks['Age'], y=open_checks[COL_AMOUNT], mode='markers',
        marker=dict(
            size=open_checks[COL_AMOUNT].apply(lambda x: np.log(x)*3 if x>0 else 5),
            color=open_checks['Age'], colorscale='Portland', showscale=True,
            line=dict(width=1, color='DarkSlateGrey')
        ),
        text=open_checks[COL_VENDOR_ID],
        hovertemplate="<b>%{text}</b><br>$%{y:,.2f}<br>%{x} days old"
    ))
    fig.update_layout(
        title="<b>Outstanding Check Landscape</b><br><sup>Color = Age | Size = Amount</sup>",
        xaxis_title="Days Since Posted", yaxis_title="Amount ($)",
        template="plotly_white", height=450
    )
    return fig

# ==========================================
# 2. MAIN APP
# ==========================================
mode = st.sidebar.radio("Navigation", ["🚀 Forecast & Intelligence", "🧪 Backtest Lab"])

if mode == "🚀 Forecast & Intelligence":
    st.header("🚀 Daily Cash Forecast")
    
    # --- UPDATE LEDGER FEATURE ---
    with st.expander("🔄 Update Ledger History (Add Cleared Checks)", expanded=False):
        st.caption("Upload a file with recently cleared checks. The system will remove duplicates and add new ones to history.")
        update_file = st.file_uploader("Upload Cleared Checks File", type=['xlsx', 'csv'], key="update_upl")
        
        if update_file and st.button("Merge into History"):
            with st.spinner("Merging..."):
                try:
                    # 1. Load the new file
                    new_df = pd.read_excel(update_file) if update_file.name.endswith('.xlsx') else pd.read_csv(update_file)
                    new_df = smart_normalize_columns(new_df)
                    
                    # 2. Load the existing Master Ledger (assuming it sits in data/master.xlsx)
                    # NOTE: Adjust path to where your master file actually lives
                    master_path = "data/master_ledger.xlsx"
                    if os.path.exists(master_path):
                        master_df = pd.read_excel(master_path)
                    else:
                        master_df = pd.DataFrame()
                        
                    if not master_df.empty:
                        master_df = smart_normalize_columns(master_df)

                    # 3. Deduplicate
                    if COL_CHECK_ID in new_df.columns and COL_CHECK_ID in master_df.columns:
                        existing_ids = set(master_df[COL_CHECK_ID].astype(str))
                        
                        # Filter for rows where Check ID is NOT in existing
                        unique_new = new_df[~new_df[COL_CHECK_ID].astype(str).isin(existing_ids)]
                        
                        if not unique_new.empty:
                            # Append
                            updated_master = pd.concat([master_df, unique_new], ignore_index=True)
                            updated_master.to_excel(master_path, index=False)
                            st.success(f"✅ Success! Added {len(unique_new)} new cleared checks to history.")
                        else:
                            st.info("ℹ️ No new checks found. All checks in file already exist in history.")
                    else:
                        # Fallback if starting fresh
                        new_df.to_excel(master_path, index=False)
                        st.success(f"✅ Created new Master Ledger with {len(new_df)} rows.")
                        
                except Exception as e:
                    st.error(f"Error updating ledger: {e}")

    # --- FORECAST INPUTS ---
    with st.expander("📂 Run Daily Forecast", expanded=True):
        c1, c2 = st.columns(2)
        run_date = c1.date_input("Run Date", value="today")
        run_date_pd = pd.to_datetime(run_date)
        uncleared_file = c2.file_uploader("Outstanding Checks (Required)", type=['xlsx', 'csv'])
        
        if c2.button("RUN FORECAST", type="primary"):
            if uncleared_file:
                with st.spinner("Processing..."):
                    os.makedirs(f"data/raw/{run_date}", exist_ok=True)
                    with open(f"data/raw/{run_date}/uncleared_checks.xlsx", "wb") as f: f.write(uncleared_file.getbuffer())
                    
                    ledger = ingest_and_reconcile(str(run_date), run_date_pd)
                    
                    if COL_VENDOR_ID not in ledger.columns or (ledger[COL_VENDOR_ID] == 'Unknown_Vendor').all():
                        ledger = smart_normalize_columns(ledger)

                    st.session_state['ledger'] = ledger
                    overrides = load_vendor_overrides()
                    engine = ForecastEngine(ledger, overrides)
                    
                    open_checks = ledger[ledger[COL_STATUS] == STATUS_OPEN].copy()
                    data = []
                    
                    # --- CLEAN PREDICTION LOOP (ENGINE HANDLES FALLBACK) ---
                    for _, row in open_checks.iterrows():
                        # The Engine handles the "Specific -> Global" fallback internally now.
                        prob = engine.predict_check(row, run_date_pd, current_date_override=run_date_pd-timedelta(days=1))
                        data.append({**row, 'Probability': prob, 'Expected_Cash': row[COL_AMOUNT]*prob})
                    
                    st.session_state['forecast_df'] = pd.DataFrame(data)
                    st.success("Done!")

    if st.session_state['forecast_df'] is not None:
        df = st.session_state['forecast_df']
        ledger = st.session_state['ledger']
        
        st.markdown("### 📊 Portfolio Overview")
        c1, c2 = st.columns([1, 3])
        with c1:
            st.metric("CASH REQUIRED TODAY", f"${df['Expected_Cash'].sum():,.2f}")
            st.metric("TOTAL EXPOSURE", f"${df[COL_AMOUNT].sum():,.2f}")
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer) as writer: df.to_excel(writer, index=False)
            st.download_button("📥 Download Excel", buffer.getvalue(), f"forecast_{run_date}.xlsx")
        with c2:
            st.plotly_chart(plot_interactive_landscape(ledger, run_date_pd), use_container_width=True)

        st.markdown("---")
        st.subheader("🕵️ Vendor Intelligence")
        
        valid_vendors = sorted([v for v in df[COL_VENDOR_ID].unique() if v != "Unknown_Vendor"])
        if not valid_vendors: valid_vendors = ["Unknown_Vendor"]
        
        sel_vendor = st.selectbox("Select Vendor:", valid_vendors)
        
        overrides = load_vendor_overrides()
        engine = ForecastEngine(ledger, overrides)
        
        if sel_vendor:
            c1, c2 = st.columns(2)
            with c1:
                if sel_vendor in engine.models['SPECIFIC']:
                    st.plotly_chart(plot_vendor_history_legacy(engine.models['SPECIFIC'][sel_vendor], sel_vendor, ledger), use_container_width=True)
                else:
                    st.info("Insufficient history for this vendor.")
            
            with c2:
                delay_days = st.slider(f"Simulate Delay for {sel_vendor}", 0, 14, 2)
                v_open = ledger[(ledger[COL_STATUS]==STATUS_OPEN) & (ledger[COL_VENDOR_ID]==sel_vendor)].copy()
                if not v_open.empty:
                    st.plotly_chart(plot_snowball_interactive(engine, v_open, run_date_pd, custom_delay=delay_days), use_container_width=True)
                    st.info(f"**Insight:** Moving the slider shows how cash flow shifts if {sel_vendor} is delayed.")
                    with st.expander("See Raw Data"):
                        v_open['Age'] = (run_date_pd - v_open[COL_POST_DATE]).dt.days
                        st.dataframe(v_open[[COL_CHECK_ID, COL_AMOUNT, COL_POST_DATE, 'Age']].sort_values('Age', ascending=False), use_container_width=True)
                else:
                    st.success("No open checks.")

# ------------------------------------------
# TAB 2: BACKTEST LAB
# ------------------------------------------
elif mode == "🧪 Backtest Lab":
    st.title("🧪 Backtest Lab")
    
    with st.expander("⚙️ Configuration", expanded=True):
        c1, c2 = st.columns(2)
        start_dt = c1.date_input("Start Date", value=datetime.today()-timedelta(days=90))
        end_dt = c1.date_input("End Date", value=datetime.today()-timedelta(days=1))
        hist_file = c2.file_uploader("Upload Master History", type=['xlsx'])

    if hist_file and st.button("RUN GLOBAL BACKTEST", type="primary"):
        with st.spinner("Processing History..."):
            try:
                raw = pd.read_excel(hist_file)
                cols_str = "".join([str(c) for c in raw.columns])
                if "Unnamed" in cols_str or raw.shape[1] < 3:
                    raw = pd.read_excel(hist_file, header=1)
            except Exception as e:
                st.error(f"Error reading file: {e}")
                st.stop()

            clean = smart_normalize_columns(raw)
            if COL_POST_DATE in clean.columns:
                clean[COL_POST_DATE] = pd.to_datetime(clean[COL_POST_DATE], dayfirst=True, errors='coerce')
            if 'Clear_Date' in clean.columns:
                clean['Clear_Date'] = pd.to_datetime(clean['Clear_Date'], dayfirst=True, errors='coerce')
            
            # --- CALCULATE DAYS_TO_SETTLE FOR ENGINE TRAINING ---
            if 'Clear_Date' in clean.columns and COL_POST_DATE in clean.columns:
                clean['Days_to_Settle'] = (clean['Clear_Date'] - clean[COL_POST_DATE]).dt.days
                # Ensure status is cleared where appropriate
                mask_cleared = clean['Clear_Date'].notna()
                clean.loc[mask_cleared, 'Status'] = 'CLEARED'
            
            st.session_state['bt_clean'] = clean
            st.session_state['bt_dates'] = (start_dt, end_dt)
            
            overrides = load_vendor_overrides()
            res = run_walk_forward_backtest(clean, str(start_dt), str(end_dt), overrides)
            st.session_state['bt_res'] = res
            st.success("Global Backtest Complete.")

    if st.session_state['bt_res'] is not None:
        res = st.session_state['bt_res']
        
        st.markdown("### 🌍 Global Results")
        m1, m2, m3 = st.columns(3)
        act = res['Actual'].sum()
        pred = res['Predicted'].sum()
        var = pred - act
        m1.metric("Actual", f"${act:,.0f}")
        m2.metric("Predicted", f"${pred:,.0f}")
        m3.metric("Variance", f"${var:,.0f}", delta=f"{(var/act)*100:.1f}%")
        
        st.plotly_chart(plot_backtest_results(res, "Global"), use_container_width=True)
        
        st.markdown("---")
        st.subheader("🔬 Vendor Drill-Down")
        clean_df = st.session_state.get('bt_clean')
        
        if clean_df is not None:
            vendors = sorted([v for v in clean_df[COL_VENDOR_ID].astype(str).unique() if v != "Unknown_Vendor"])
            if not vendors: vendors = ["Unknown_Vendor"]

            c_sel, c_btn = st.columns([3, 1])
            v_sel = c_sel.selectbox("Select Vendor:", vendors)
            
            if c_btn.button(f"Run for {v_sel}"):
                s, e = st.session_state['bt_dates']
                overrides = load_vendor_overrides()
                with st.spinner(f"Simulating {v_sel}..."):
                    v_res = run_walk_forward_backtest(clean_df, str(s), str(e), overrides, vendor_filter=v_sel)
                    if not v_res.empty:
                        v_act = v_res['Actual'].sum()
                        v_pred = v_res['Predicted'].sum()
                        v_var = v_pred - v_act
                        c1, c2, c3 = st.columns(3)
                        c1.metric(f"{v_sel} Actual", f"${v_act:,.0f}")
                        c2.metric(f"{v_sel} Predicted", f"${v_pred:,.0f}")
                        c3.metric("Variance", f"${v_var:,.0f}")
                        st.plotly_chart(plot_backtest_results(v_res, v_sel), use_container_width=True)
                    else:
                        st.warning("No cleared checks found for this vendor in this period.")
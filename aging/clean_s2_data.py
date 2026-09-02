import os
import csv
import argparse

parser = argparse.ArgumentParser(description="Clean S2 battery data.")
parser.add_argument("--raw_dir", type=str, default="data", help="Directory containing raw csv files")
parser.add_argument("--out_dir", type=str, default="output", help="Directory to save cleaned csv files")
args = parser.parse_args()

raw_dir = args.raw_dir
out_dir = args.out_dir

os.makedirs(out_dir, exist_ok=True)

files = [
    "2026-05-03_LFP280_A1.csv",
    "2026-05-18_LFP280_A2.csv",
    "0602_測試檔_A3.csv",
    "2026-06-25_LFP280_B1_修正版.csv"
]

target_header = [
    "cell_id", "model", "cycle_index", "start_time", "end_time",
    "channel", "chg_rate_c", "dchg_rate_c", "v_upper_v", "v_lower_v",
    "avg_temp_c", "chg_cap_ah", "dchg_cap_ah", "source_file"
]

def parse_time(val):
    if not val or not val.strip():
        return ""
    val = val.strip()
    val = val.replace("年", "-").replace("月", "-").replace("日", "")
    val = val.replace("/", "-")
    parts = val.split(" ")
    date_part = parts[0]
    time_part = parts[1] if len(parts) > 1 else "00:00:00"
    
    d_items = date_part.split("-")
    year = d_items[0].zfill(4)
    month = d_items[1].zfill(2)
    day = d_items[2].zfill(2)
    
    t_items = time_part.split(":")
    hour = t_items[0].zfill(2)
    minute = t_items[1].zfill(2)
    second = t_items[2].zfill(2) if len(t_items) > 2 else "00"
    
    return f"{year}-{month}-{day} {hour}:{minute}:{second}"

def clean_cell_id(val):
    val = val.strip()
    if val == "B1":
        return "LFP280-B1"
    return val.replace("_", "-")

def parse_rate(val):
    if not val or not val.strip():
        return ""
    return val.strip().upper().replace("C", "")

master_rows = []

for fname in files:
    filepath = os.path.join(raw_dir, fname)
    out_file = os.path.join(out_dir, f"cleaned_{fname}")
    
    cleaned_rows = []
    with open(filepath, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            chg_cap = float(row["Chg_Cap"]) if row["Chg_Cap"].strip() else 0.0
            dchg_cap = float(row["DChg_Cap"]) if row["DChg_Cap"].strip() else 0.0
            
            if chg_cap > 1000:
                chg_cap /= 1000.0
            if dchg_cap > 1000:
                dchg_cap /= 1000.0
                
            avg_temp = row["Avg_Temp"].strip() if row["Avg_Temp"] else ""
            
            c_row = {
                "cell_id": clean_cell_id(row["Cell_ID"]),
                "model": row["Model"].strip(),
                "cycle_index": int(row["Cycle"]),
                "start_time": parse_time(row["Start Time"]),
                "end_time": parse_time(row["End Time"]),
                "channel": row["Channel"].strip(),
                "chg_rate_c": parse_rate(row["Chg_Rate"]),
                "dchg_rate_c": parse_rate(row["DChg_Rate"]),
                "v_upper_v": float(row["V_upper"]),
                "v_lower_v": float(row["V_lower"]),
                "avg_temp_c": avg_temp,
                "chg_cap_ah": f"{chg_cap:.3f}",
                "dchg_cap_ah": f"{dchg_cap:.3f}",
                "source_file": fname
            }
            cleaned_rows.append(c_row)
            master_rows.append(c_row)
            
    with open(out_file, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=target_header)
        writer.writeheader()
        writer.writerows(cleaned_rows)
    print(f"Saved: {out_file} ({len(cleaned_rows)} rows)")

master_file = os.path.join(out_dir, "cleaned_master_cycle_data.csv")
with open(master_file, "w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=target_header)
    writer.writeheader()
    writer.writerows(master_rows)
print(f"Saved Master: {master_file} ({len(master_rows)} rows)")

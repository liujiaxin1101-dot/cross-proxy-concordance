import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np

PARTICIPANT_LOG = os.path.join(
    os.environ.get("C3D_DATA_DIR", os.path.join(PROJECT_ROOT, "..", "c3d_data")),
    "Participants", "participant_log.xlsx")

df_raw = pd.read_excel(PARTICIPANT_LOG, header=None)
print("=== RAW SHAPE ===")
print(f"  {df_raw.shape}")
print()

col0 = df_raw.iloc[:, 0].astype(str)
has_sub = col0.str.match(r'^sub\d+\s*$')
data_rows = df_raw[has_sub].copy()
print(f"Data rows (subXX): {len(data_rows)}")

group_column = df_raw.iloc[0, 1]
group_col_idx = 1
subject_col_idx = 0
gender_col_idx = 2
age_col_idx = 3
height_col_idx = 4
weight_col_idx = 5

print(f"Group column name: '{group_column}'")
print()

ACL = []
Control = []
for _, row in data_rows.iterrows():
    sub_id = str(row.iloc[subject_col_idx]).strip()
    group_val = row.iloc[group_col_idx]

    try:
        group_val = int(group_val)
    except (ValueError, TypeError):
        continue

    if group_val == 1:
        ACL.append(sub_id)
    elif group_val == 2:
        Control.append(sub_id)

    gender = int(row.iloc[gender_col_idx])
    age = row.iloc[age_col_idx]
    height = row.iloc[height_col_idx]
    weight = row.iloc[weight_col_idx]
    gender_str = "F" if gender == 1 else "M"
    label = "ACL" if group_val == 1 else "CTL"
    print(f"  {sub_id}  |  {label}  |  {gender_str}  |  age={age}  |  {height}m  |  {weight}kg")

print()
print("=== SUMMARY ===")
print(f"  ACL group:    {len(ACL)} subjects")
print(f"  Control group: {len(Control)} subjects")
print()
print(f"  ACL IDs:    {sorted(ACL)}")
print(f"  Control IDs: {sorted(Control)}")
print()

ctrl_ids_csv = ", ".join(f'"{s}"' for s in sorted(Control))
print(f"  Control list for code: [{ctrl_ids_csv}]")

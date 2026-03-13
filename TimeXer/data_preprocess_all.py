import pandas as pd
import numpy as np
import os
import pathlib
import glob
from datetime import datetime, timedelta
import re
from concurrent.futures import ProcessPoolExecutor

PROCESS_BASE_DIR = pathlib.Path(__file__).parent.absolute()
SOURCE_DIR = PROCESS_BASE_DIR / "outside data" / "total bullet"
DEST_DIR = PROCESS_BASE_DIR / "dataset" / "bullet"
SOLO_DIR = DEST_DIR / "solo"
CATEGORY_DIR = DEST_DIR / "category"

EXO_BASE_DIR = str(PROCESS_BASE_DIR / "dataset")
HOLIDAY_PATH = os.path.join(EXO_BASE_DIR, "China_Holiday_2025_2026.csv")
SEASON_PATH = os.path.join(EXO_BASE_DIR, "赛季时间.csv")
GUN_EVENT_PATH = os.path.join(EXO_BASE_DIR, "品枪时间.csv")
PREDICTION_PATH = os.path.join(EXO_BASE_DIR, "predictions.csv")

EXO_CATEGORY_DIR = os.path.join(EXO_BASE_DIR, "bullet", "category")
EXO_SOLO_DIR = os.path.join(EXO_BASE_DIR, "bullet", "solo")
OUT_CATEGORY_DIR = os.path.join(EXO_BASE_DIR, "bullet", "collection_category")
OUT_SOLO_DIR = os.path.join(EXO_BASE_DIR, "bullet", "collection_solo")

ALIAS_MAP = {
    "arrow 3": "玻纤柳叶箭矢",
    "玻纤柳叶箭矢": "arrow 3",
    "arrow 4": "碳纤维刺骨箭矢",
    "碳纤维刺骨箭矢": "arrow 4",
    "arrow 5": "碳纤维穿甲箭矢",
    "碳纤维穿甲箭矢": "arrow 5"
}

def get_category(filename):
    stem = filename.replace('.csv', '')
    if stem.endswith("箭矢"):
        return "箭矢"
    special_prefixes = [
        '.357 Magnum',
        '.45 ACP',
        '.50 AE',
        '12 Gauge',
        '45-70 Govt'
    ]
    for prefix in special_prefixes:
        if stem.startswith(prefix):
            return prefix
    return stem.split(' ')[0]

def process_data():
    print(f"Processing data from {SOURCE_DIR}...")
    os.makedirs(SOLO_DIR, exist_ok=True)
    os.makedirs(CATEGORY_DIR, exist_ok=True)
    category_map = {}
    files = [f for f in os.listdir(SOURCE_DIR) if f.endswith('.csv')]
    for file in files:
        file_path = SOURCE_DIR / file
        try:
            df = pd.read_csv(file_path)
            if '时间' not in df.columns or '均价' not in df.columns:
                print(f"Skipping {file}: Missing required columns '时间' or '均价'")
                continue
            df_solo = df[['时间', '均价']].copy()
            df_solo.rename(columns={'时间': 'date', '均价': 'OT'}, inplace=True)
            solo_output_path = SOLO_DIR / file
            df_solo.to_csv(solo_output_path, index=False, encoding='utf-8-sig')
            category = get_category(file)
            col_name = file.replace('.csv', '')
            df_multi = df_solo.copy()
            df_multi.rename(columns={'OT': col_name}, inplace=True)
            if category not in category_map:
                category_map[category] = []
            category_map[category].append(df_multi)
        except Exception as e:
            print(f"Error processing {file}: {e}")
    print("Generating multivariate category files...")
    for category, dfs in category_map.items():
        if not dfs:
            continue
        merged_df = dfs[0]
        for next_df in dfs[1:]:
            merged_df = pd.merge(merged_df, next_df, on='date', how='outer')
        merged_df['date'] = pd.to_datetime(merged_df['date'])
        merged_df = merged_df.sort_values('date')
        merged_df = merged_df.reset_index(drop=True)
        for col in merged_df.columns:
            if col == 'date':
                continue
            first_valid_idx = merged_df[col].first_valid_index()
            if first_valid_idx is None or first_valid_idx == 0:
                continue
            fill_value = merged_df.at[first_valid_idx, col]
            merged_df.loc[: first_valid_idx - 1, col] = fill_value
        output_path = CATEGORY_DIR / f"{category}.csv"
        merged_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"Created {category}.csv with {len(merged_df.columns) - 1} variables.")
    print("Processing complete.")

def get_aliases(name):
    names = {name}
    if name in ALIAS_MAP:
        names.add(ALIAS_MAP[name])
    return names

def parse_chinese_date(date_str):
    try:
        match = re.match(r"(\d+)年(\d+)月(\d+)日", str(date_str))
        if match:
            return datetime(int(match.group(1)), int(match.group(2)), int(match.group(3)))
    except:
        pass
    return None

def load_configs():
    print("Loading configs...")
    try:
        holidays_df = pd.read_csv(HOLIDAY_PATH)
        holiday_map = {}
        for _, row in holidays_df.iterrows():
            try:
                d = pd.to_datetime(row['date']).date()
                holiday_map[d] = int(row['is_holiday'])
            except:
                pass
    except Exception as e:
        print(f"Warning: Could not load holidays: {e}")
        holiday_map = {}
    seasons = []
    try:
        seasons_df = pd.read_csv(SEASON_PATH)
        for _, row in seasons_df.iterrows():
            s_name = str(row['赛季名称'])
            s_id = 0
            if 's6' in s_name.lower(): s_id = 1
            if 's7' in s_name.lower(): s_id = 2
            start_date = parse_chinese_date(row['赛季开始'])
            end_date_raw = parse_chinese_date(row['赛季结束'])
            if start_date and end_date_raw:
                end_date = end_date_raw + timedelta(days=1)
                needed = str(row['赛季任务所需子弹']).split(';') if pd.notna(row['赛季任务所需子弹']) else []
                made = str(row['制造子弹']).split(';') if pd.notna(row['制造子弹']) else []
                seasons.append({
                    'id': s_id,
                    'start': start_date,
                    'end': end_date,
                    'needed': [x.strip() for x in needed if x.strip()],
                    'made': [x.strip() for x in made if x.strip()]
                })
    except Exception as e:
        print(f"Warning: Could not load seasons: {e}")
    gun_events = []
    try:
        gun_events_df = pd.read_csv(GUN_EVENT_PATH)
        for _, row in gun_events_df.iterrows():
            time_range = str(row['时间'])
            bullets = str(row['所用子弹'])
            try:
                start_str, _ = time_range.split('-')
                def parse_md(s):
                    m, d = map(int, s.split('.'))
                    y = 2025 if m >= 9 else 2026
                    return datetime(y, m, d)
                s_date = parse_md(start_str)
                active_start = s_date - timedelta(days=2)
                active_end = s_date + timedelta(days=2)
                active_end_exclusive = active_end + timedelta(days=1)
                gun_events.append({
                    'start': active_start,
                    'end': active_end_exclusive,
                    'bullet': bullets.strip()
                })
            except Exception as e:
                pass
    except Exception as e:
        print(f"Warning: Could not load gun events: {e}")
    predictions = []
    try:
        pred_df = pd.read_csv(PREDICTION_PATH)
        for _, row in pred_df.iterrows():
            try:
                t = pd.to_datetime(row['time'])
                n = str(row['name'])
                predictions.append({'time': t, 'name': n})
            except:
                pass
    except Exception as e:
        print(f"Warning: Could not load predictions: {e}")
    return holiday_map, seasons, gun_events, predictions

def get_is_holiday(dt, holiday_map):
    d = dt.date()
    return holiday_map.get(d, 0)

def get_season_data(dt, bullet_name, seasons):
    in_cs = 0.0
    is_cs = 0
    is_need = 0
    is_make = 0
    aliases = get_aliases(bullet_name)
    for s in seasons:
        if s['start'] <= dt < s['end']:
            total_duration = (s['end'] - s['start']).total_seconds()
            elapsed = (dt - s['start']).total_seconds()
            val = elapsed / total_duration if total_duration > 0 else 0
            val = max(0.0, min(1.0, val))
            in_cs = val
            is_cs = s['id']
            for nb in s['needed']:
                for alias in aliases:
                    if nb in alias:
                        is_need = 1
                        break
                if is_need: break
            for mb in s['made']:
                for alias in aliases:
                    if mb in alias:
                        is_make = 1
                        break
                if is_make: break
            break
    return in_cs, is_cs, is_need, is_make

def get_is_active(dt, bullet_name, gun_events):
    aliases = get_aliases(bullet_name)
    for ev in gun_events:
        if ev['start'] <= dt < ev['end']:
            for alias in aliases:
                if ev['bullet'] in alias:
                    return 1
    return 0

def get_is_public(dt, bullet_name, predictions):
    val = 0.0
    aliases = get_aliases(bullet_name)
    for p in predictions:
        match = False
        for alias in aliases:
            if p['name'] in alias:
                match = True
                break
        if match:
            delta = dt - p['time']
            days_diff = delta.total_seconds() / (24 * 3600)
            if 0 <= days_diff <= 60:
                curr_val = 1.0 - (days_diff / 60.0)
                if curr_val > val:
                    val = curr_val
    return val

def process_single_file(args):
    file_path, dest_path, holiday_map, seasons, gun_events, predictions = args
    try:
        df = pd.read_csv(file_path)
        bullet_name = os.path.splitext(os.path.basename(file_path))[0]
        df['date'] = pd.to_datetime(df['date'])
        holidays_col = []
        in_cs_col = []
        is_cs_col = []
        is_need_col = []
        is_make_col = []
        is_active_col = []
        is_public_col = []
        for dt in df['date']:
            holidays_col.append(get_is_holiday(dt, holiday_map))
            inc, isc, isn, ism = get_season_data(dt, bullet_name, seasons)
            in_cs_col.append(inc)
            is_cs_col.append(isc)
            is_need_col.append(isn)
            is_make_col.append(ism)
            is_active_col.append(get_is_active(dt, bullet_name, gun_events))
            is_public_col.append(get_is_public(dt, bullet_name, predictions))
        df['is_holiday'] = holidays_col
        df['in_CS'] = in_cs_col
        df['is_CS'] = is_cs_col
        df['is_need'] = is_need_col
        df['is_make'] = is_make_col
        df['is_active'] = is_active_col
        df['is_public'] = is_public_col
        df.to_csv(dest_path, index=False, encoding='utf-8')
        return None
    except Exception as e:
        return f"Failed {os.path.basename(file_path)}: {e}"

def add_exogenous_variables_main():
    os.makedirs(OUT_CATEGORY_DIR, exist_ok=True)
    os.makedirs(OUT_SOLO_DIR, exist_ok=True)
    holiday_map, seasons, gun_events, predictions = load_configs()
    tasks = []
    for f in os.listdir(EXO_CATEGORY_DIR):
        if f.endswith(".csv"):
            full_path = os.path.join(EXO_CATEGORY_DIR, f)
            dest = os.path.join(OUT_CATEGORY_DIR, f)
            tasks.append((full_path, dest, holiday_map, seasons, gun_events, predictions))
    for f in os.listdir(EXO_SOLO_DIR):
        if f.endswith(".csv"):
            full_path = os.path.join(EXO_SOLO_DIR, f)
            dest = os.path.join(OUT_SOLO_DIR, f)
            tasks.append((full_path, dest, holiday_map, seasons, gun_events, predictions))
    print(f"Found {len(tasks)} files to process.")
    with ProcessPoolExecutor(max_workers=16) as executor:
        results = list(executor.map(process_single_file, tasks))
    failures = [r for r in results if r]
    if failures:
        print("Failures:")
        for f in failures:
            print(f)
    else:
        print("All files processed successfully.")

def main():
    process_data()
    add_exogenous_variables_main()

if __name__ == "__main__":
    main()

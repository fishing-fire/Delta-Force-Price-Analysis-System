import os
import re
import json
import pandas as pd

pattern = re.compile(r'^(.*)_ft([^_]+)_sl(\d+)_ll(\d+)_pl(\d+)_dm(\d+)_nh(\d+)_el(\d+)_dl(\d+)_df(\d+)_expand(\d+)_dc(\d+)_fc(\d+)_eb([^_]+)_dt([^_]+)_(.*)_(\d+)$')

results_dir = 'results'
output_file = 'summary_metrics.csv'

data_list = []

# 已知的任务名称列表，用于辅助解析前缀
KNOWN_TASKS = [
    'long_term_forecast',
    'short_term_forecast',
    'imputation',
    'classification',
    'anomaly_detection'
]

def parse_prefix(prefix):
    """
    解析前缀部分，提取 Task Name, Model ID, Model Name, Data
    假设结构为: {Task}_{ModelID}_{Model}_{Data}
    其中 Task 是已知的，Model 和 Data 不含下划线（或者作为最后两个部分）
    """
    parts = prefix.split('_')
    
    # 1. 识别 Task Name
    task_name = "Unknown"
    remaining_parts = parts
    
    for task in KNOWN_TASKS:
        if prefix.startswith(task):
            task_name = task
            task_len = len(task.split('_'))
            remaining_parts = parts[task_len:]
            break
            
    if not remaining_parts:
        return task_name, "Unknown", "Unknown", "Unknown"
        
    # 2. 识别 Model 和 Data (假设是最后两个)
    if len(remaining_parts) >= 2:
        data = remaining_parts[-1]
        model = remaining_parts[-2]
        model_id = "_".join(remaining_parts[:-2]) if len(remaining_parts) > 2 else ""
        if not model_id:
             model_id = "Default" # 如果中间没有内容了
    elif len(remaining_parts) == 1:
        data = remaining_parts[0]
        model = "Unknown"
        model_id = "Default"
    else:
        data = "Unknown"
        model = "Unknown"
        model_id = "Unknown"
        
    return task_name, model_id, model, data

def read_meta_from_log(log_path):
    if not log_path or not os.path.exists(log_path):
        return None
    try:
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            for _ in range(500):
                line = f.readline()
                if not line:
                    break
                line = line.lstrip('\ufeff').strip()
                if line.startswith('META:'):
                    payload = line[len('META:'):].strip()
                    return json.loads(payload)
    except Exception:
        return None
    return None

def _normalize_bool(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ('true', '1', 'yes', 'y', 'on'):
            return True
        if v in ('false', '0', 'no', 'n', 'off'):
            return False
    return value

def extract_params_from_meta(meta, model_id_fallback=None):
    if not meta:
        return None
    params = {
        'Model Name': meta.get('model', 'Unknown'),
        'Task Name': meta.get('task_name', 'Unknown'),
        'Model ID': meta.get('model_id', model_id_fallback if model_id_fallback is not None else 'Unknown'),
        'Data': meta.get('data', 'Unknown'),
        'Features': meta.get('features', None),
        'Seq Len': meta.get('seq_len', None),
        'Label Len': meta.get('label_len', None),
        'Pred Len': meta.get('pred_len', None),
        'D Model': meta.get('d_model', None),
        'N Heads': meta.get('n_heads', None),
        'E Layers': meta.get('e_layers', None),
        'D Layers': meta.get('d_layers', None),
        'D FF': meta.get('d_ff', None),
        'Expand': meta.get('expand', None),
        'D Conv': meta.get('d_conv', None),
        'Factor': meta.get('factor', None),
        'Embed': meta.get('embed', None),
        'Distil': _normalize_bool(meta.get('distil', None)),
        'Description': meta.get('des', None),
        'Iteration': meta.get('itr_index', None),
    }
    return params

def read_metrics(metrics_path):
    metrics_df = pd.read_csv(metrics_path, nrows=1)
    if metrics_df.empty:
        return None
    return {
        'MAE': metrics_df['MAE'].iloc[0] if 'MAE' in metrics_df.columns else None,
        'MSE': metrics_df['MSE'].iloc[0] if 'MSE' in metrics_df.columns else None,
        'RMSE': metrics_df['RMSE'].iloc[0] if 'RMSE' in metrics_df.columns else None,
        'MAPE': metrics_df['MAPE'].iloc[0] if 'MAPE' in metrics_df.columns else None,
        'MSPE': metrics_df['MSPE'].iloc[0] if 'MSPE' in metrics_df.columns else None,
    }

def _parse_legacy_dirname(folder_name):
    match = pattern.match(folder_name)
    if not match:
        return None

    prefix = match.group(1)
    ft = match.group(2)
    sl = match.group(3)
    ll = match.group(4)
    pl = match.group(5)
    dm = match.group(6)
    nh = match.group(7)
    el = match.group(8)
    dl = match.group(9)
    df = match.group(10)
    expand = match.group(11)
    dc = match.group(12)
    fc = match.group(13)
    eb = match.group(14)
    dt = match.group(15)
    des = match.group(16)
    ii = match.group(17)

    task_name, model_id, model, data = parse_prefix(prefix)
    return {
        'Model Name': model,
        'Task Name': task_name,
        'Model ID': model_id,
        'Data': data,
        'Features': ft,
        'Seq Len': int(sl) if str(sl).isdigit() else sl,
        'Label Len': int(ll) if str(ll).isdigit() else ll,
        'Pred Len': int(pl) if str(pl).isdigit() else pl,
        'D Model': int(dm) if str(dm).isdigit() else dm,
        'N Heads': int(nh) if str(nh).isdigit() else nh,
        'E Layers': int(el) if str(el).isdigit() else el,
        'D Layers': int(dl) if str(dl).isdigit() else dl,
        'D FF': int(df) if str(df).isdigit() else df,
        'Expand': int(expand) if str(expand).isdigit() else expand,
        'D Conv': int(dc) if str(dc).isdigit() else dc,
        'Factor': int(fc) if str(fc).isdigit() else fc,
        'Embed': eb,
        'Distil': _normalize_bool(dt),
        'Description': des,
        'Iteration': int(ii) if str(ii).isdigit() else ii,
    }

def process_experiment_dir(exp_dir_path, fallback_dirname, model_id_fallback):
    metrics_path = os.path.join(exp_dir_path, 'metrics.csv')
    if not os.path.exists(metrics_path):
        return None
    metrics = read_metrics(metrics_path)
    if not metrics:
        return None

    meta = read_meta_from_log(os.path.join(exp_dir_path, 'log.txt'))
    params = extract_params_from_meta(meta, model_id_fallback=model_id_fallback) if meta else None

    if not params:
        legacy_params = _parse_legacy_dirname(fallback_dirname)
        if legacy_params:
            params = legacy_params
        else:
            params = {
                'Model Name': 'Unknown',
                'Task Name': 'Unknown',
                'Model ID': model_id_fallback if model_id_fallback is not None else fallback_dirname,
                'Data': 'Unknown',
                'Features': None,
                'Seq Len': None,
                'Label Len': None,
                'Pred Len': None,
                'D Model': None,
                'N Heads': None,
                'E Layers': None,
                'D Layers': None,
                'D FF': None,
                'Expand': None,
                'D Conv': None,
                'Factor': None,
                'Embed': None,
                'Distil': None,
                'Description': None,
                'Iteration': None,
            }

    if model_id_fallback is not None:
        params['Model ID'] = model_id_fallback

    params.update(metrics)
    return params

if not os.path.exists(results_dir):
    print(f"Directory {results_dir} does not exist.")
    exit(1)

print(f"Scanning directory: {results_dir}...")

for folder_name in os.listdir(results_dir):
    folder_path = os.path.join(results_dir, folder_name)
    if not os.path.isdir(folder_path):
        continue

    entry = process_experiment_dir(folder_path, fallback_dirname=folder_name, model_id_fallback=folder_name)
    if entry:
        data_list.append(entry)

    for run_dir_name in os.listdir(folder_path):
        run_dir_path = os.path.join(folder_path, run_dir_name)
        if not os.path.isdir(run_dir_path):
            continue
        if not os.path.exists(os.path.join(run_dir_path, 'metrics.csv')):
            continue
        entry = process_experiment_dir(run_dir_path, fallback_dirname=folder_name, model_id_fallback=run_dir_name)
        if entry:
            data_list.append(entry)

# 创建 DataFrame 并保存
if data_list:
    df_result = pd.DataFrame(data_list)
    
    # 调整列顺序，让指标排在最后
    cols = [
        'Model Name', 'Task Name', 'Model ID',
        'MAE', 'MSE', 'RMSE', 'MAPE', 'MSPE',
        'Data', 'Features', 'Seq Len', 'Label Len', 'Pred Len',
        'D Model', 'N Heads', 'E Layers', 'D Layers', 'D FF',
        'Expand', 'D Conv', 'Factor', 'Embed', 'Distil',
        'Description', 'Iteration'
    ]
    # 确保只包含存在的列
    cols = [c for c in cols if c in df_result.columns]
    df_result = df_result[cols]
    
    df_result.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\nSuccessfully collected metrics from {len(df_result)} experiments.")
    print(f"Saved to: {os.path.abspath(output_file)}")
else:
    print("\nNo valid experiment results found.")

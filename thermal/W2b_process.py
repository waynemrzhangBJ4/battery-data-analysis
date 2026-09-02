import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, cumulative_trapezoid
from scipy.optimize import least_squares

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', default='data')
parser.add_argument('--out-dir', default='output')
args = parser.parse_args()

data_dir = args.data_dir
out_dir = args.out_dir
os.makedirs(out_dir, exist_ok=True)

# Fixed Constants (§1)
TAU_A = 841
SIGMA_A = 2.02
T_AMB_BASE = 24.0
SIGMA_B_OCV = 91.67
DTAU_PER_10MV_S = 25.5975
HA_PATH_A = 0.0499

meta_df = pd.read_csv(os.path.join(data_dir, 'metadata.csv'))
b0005 = meta_df[meta_df['battery_id'] == 'B0005'].copy()

# Pairing Logic (from W2pre_b)
def parse_time(t_str):
    if pd.isna(t_str): return tuple()
    clean = t_str.replace('[', '').replace(']', '').replace(',', '')
    return tuple(float(x) for x in clean.split())

b0005['pt'] = b0005['start_time'].apply(parse_time)
b0005 = b0005.sort_values('uid')

excluded_points = []
failures = []
valid_pairs = []

b0005_list = b0005.to_dict('records')
idx = 0
pairs = []

while idx < len(b0005_list):
    row = b0005_list[idx]
    if row['type'] == 'discharge':
        c_idx = idx + 1
        found_charge = None
        while c_idx < len(b0005_list):
            if b0005_list[c_idx]['type'] == 'charge':
                found_charge = b0005_list[c_idx]
                break
            elif b0005_list[c_idx]['type'] == 'impedance':
                c_idx += 1
            else:
                break
        if found_charge:
            pairs.append((row, found_charge))
        idx = c_idx
    else:
        idx += 1

def log_failure(uid, stage, error_type, message):
    failures.append({
        'uid': uid, 'stage': stage, 'error_type': error_type, 'message': message
    })

def get_data(uid, filename, current_thresh, is_charge=False):
    df = pd.read_csv(os.path.join(data_dir, 'data', filename))
    bad = df[df['Current_measured'].abs() > current_thresh]
    for r_idx, row in bad.iterrows():
        excluded_points.append({
            'uid': uid,
            'row_index': r_idx,
            'Time': row['Time'],
            'Current_measured': row['Current_measured'],
            'reason': f'|I| > {current_thresh}'
        })
    df = df[df['Current_measured'].abs() <= current_thresh].copy()
    if is_charge:
        # For charge E_ocp building, include CC and CV: Current_measured > 0.02
        df = df[df['Current_measured'] > 0.02].copy()
    return df

n_unpaired = len(b0005[b0005['type']=='discharge']) - len(pairs)
n_paired = len(pairs)

for d_meta, c_meta in pairs:
    uid_dis = d_meta['uid']
    uid_chg = c_meta['uid']
    
    # §3.1 Exclude transients
    d_df = get_data(uid_dis, d_meta['filename'], 2.3, is_charge=False)
    c_df = get_data(uid_chg, c_meta['filename'], 2.0, is_charge=True)
    
    if len(d_df) == 0 or len(c_df) == 0:
        log_failure(uid_dis, 'pairing', 'no_data', 'Empty dataframe after filtering')
        continue
        
    d_time = d_df['Time'].values
    d_i = np.abs(d_df['Current_measured'].values)
    d_q = cumulative_trapezoid(d_i, d_time, initial=0) / 3600.0
    d_df['Q_fromfull'] = d_q
    Q_dis_total = d_q[-1]
    
    c_df_raw = pd.read_csv(os.path.join(data_dir, 'data', c_meta['filename']))
    c_time_raw = c_df_raw['Time'].values
    c_i_raw = c_df_raw['Current_measured'].values
    c_q_int = cumulative_trapezoid(c_i_raw, c_time_raw, initial=0) / 3600.0
    Q_chg_total = c_q_int[-1]
    c_df['Q_fromfull'] = Q_chg_total - np.interp(c_df['Time'].values, c_time_raw, c_q_int)
    
    # Pair valid rules (V1, V2)
    c_raw_mask = (c_df_raw['Voltage_measured'] > 4.19) & (c_df_raw['Current_measured'] > 0.02) & (c_df_raw['Current_measured'] <= 1.4)
    has_cv = c_raw_mask.any()
    last_5_i = c_df_raw['Current_measured'].iloc[-5:].median()
    fail_V1 = not (has_cv and last_5_i < 0.05)
    fail_V2 = abs(Q_chg_total - Q_dis_total) / Q_dis_total >= 0.10
    
    if fail_V1 or fail_V2:
        log_failure(uid_dis, 'pairing', 'V1_V2_fail', 'V1 or V2 failed')
        continue
        
    valid_pairs.append((uid_dis, uid_chg, d_df, c_df))

print(f"Valid pairs: {len(valid_pairs)}")
# Continue W2b_process.py
cycle_grids = []
fit_results = []
compat_check = []

def solve_thermal_model(t, T0, I_t, V_t, E_ocp_t, mCp, hA, dudt, Tamb):
    T = np.zeros_like(t)
    T[0] = T0
    
    def dTdt(t_val, T_val):
        i = np.interp(t_val, t, I_t)
        v = np.interp(t_val, t, V_t)
        e = np.interp(t_val, t, E_ocp_t)
        Q_gen = i * (e - v) + i * T_val * dudt
        return (Q_gen - hA * (T_val - (Tamb + 273.15))) / mCp

    for idx in range(len(t) - 1):
        t_curr = t[idx]
        T_curr = T[idx]
        t_next = t[idx+1]
        
        dt_full = t_next - t_curr
        if dt_full <= 0:
            T[idx+1] = T_curr
            continue
            
        n_steps = max(1, int(np.ceil(dt_full / 5.0)))
        dt = dt_full / n_steps
        
        T_temp = T_curr
        for step in range(n_steps):
            t_step = t_curr + step * dt
            k1 = dTdt(t_step, T_temp)
            k2 = dTdt(t_step + dt/2.0, T_temp + dt*k1/2.0)
            k3 = dTdt(t_step + dt/2.0, T_temp + dt*k2/2.0)
            k4 = dTdt(t_step + dt, T_temp + dt*k3)
            T_temp += (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        T[idx+1] = T_temp
        
    return T

def fit_cycle(uid_dis, d_df, c_df, cycle_index, dudt=0.0, Tamb=T_AMB_BASE):
    # E_ocp Grid
    Q_d_min, Q_d_max = d_df['Q_fromfull'].min(), d_df['Q_fromfull'].max()
    Q_c_min, Q_c_max = c_df['Q_fromfull'].min(), c_df['Q_fromfull'].max()
    
    overlap_lo = max(Q_d_min, Q_c_min)
    overlap_hi = min(Q_d_max, Q_c_max)
    
    if overlap_hi <= overlap_lo:
        log_failure(uid_dis, 'e_ocp', 'no_overlap', 'No overlap between charge and discharge')
        return None
        
    grid = np.arange(np.ceil(overlap_lo/0.05)*0.05, overlap_hi, 0.05)
    if len(grid) == 0:
        log_failure(uid_dis, 'e_ocp', 'empty_grid', 'Empty grid')
        return None
        
    d_cc_s = d_df.sort_values('Q_fromfull')
    c_cc_s = c_df.sort_values('Q_fromfull')
    
    V_dis_interp = np.interp(grid, d_cc_s['Q_fromfull'], d_cc_s['Voltage_measured'])
    V_chg_interp = np.interp(grid, c_cc_s['Q_fromfull'], c_cc_s['Voltage_measured'])
    I_chg_interp = np.interp(grid, c_cc_s['Q_fromfull'], c_cc_s['Current_measured'].abs())
    I_dis = np.abs(d_cc_s['Current_measured'].median())
    
    E_weighted_grid = (I_dis * V_chg_interp + I_chg_interp * V_dis_interp) / (I_chg_interp + I_dis)
    
    for q, e, ic, vc in zip(grid, E_weighted_grid, I_chg_interp, V_chg_interp):
        if dudt == 0.0 and Tamb == T_AMB_BASE:
            cv_flag = 1.0 if (vc > 4.19 and 0.02 < ic <= 1.4) else 0.0
            cycle_grids.append({
                'uid': uid_dis, 'cycle_index': cycle_index, 'Q_fromfull': q, 'E_ocp': e,
                'chg_phase_frac_CV': cv_flag, 'n_pairs': 1
            })

    # Fit window
    # CC end depends on cut-off voltage, B0005 cut-off is ~2.7V
    cc_end_idx = d_df.index[d_df['Voltage_measured'] < 2.8].min()
    if pd.isna(cc_end_idx):
        fit_df = d_df
    else:
        fit_df = d_df.loc[:cc_end_idx]
        
    T0_meas = fit_df.iloc[0]['Temperature_measured']
    T0_K = T0_meas + 273.15
    
    t_vals = fit_df['Time'].values
    T_meas = fit_df['Temperature_measured'].values + 273.15
    I_vals = np.abs(fit_df['Current_measured'].values)
    V_vals = fit_df['Voltage_measured'].values
    Q_vals = fit_df['Q_fromfull'].values
    
    n_out_of_grid = np.sum((Q_vals < grid[0]) | (Q_vals > grid[-1]))
    
    # We must only interpolate within the grid bounds. If outside, use nearest or linear extrapolation.
    # Spec says "放電點落在該循環 E_ocp 網格範圍外 → 剔除該點並計數（檢查 X1）。"
    # Oh wait! "剔除該點並計數". So we must exclude those points from fitting.
    in_grid_mask = (Q_vals >= grid[0]) & (Q_vals <= grid[-1])
    if in_grid_mask.sum() < 30:
        log_failure(uid_dis, 'fit', 'too_few_points', f'Points in grid: {in_grid_mask.sum()}')
        return None
        
    t_vals = t_vals[in_grid_mask]
    T_meas = T_meas[in_grid_mask]
    I_vals = I_vals[in_grid_mask]
    V_vals = V_vals[in_grid_mask]
    Q_vals = Q_vals[in_grid_mask]
    
    E_ocp_vals = np.interp(Q_vals, grid, E_weighted_grid)
    n_negQ = np.sum((E_ocp_vals - V_vals) < 0)
    
    def residuals(params):
        mCp, hA = params
        T_model = solve_thermal_model(t_vals, T0_K, I_vals, V_vals, E_ocp_vals, mCp, hA, dudt, Tamb)
        return T_model - T_meas
        
    starts = [[40.0, 0.05], [20.0, 0.02], [60.0, 0.08]]
    ls_results = []
    taus = []
    
    for s0 in starts:
        try:
            r_opt = least_squares(residuals, s0, bounds=([5, 0.005], [150, 0.5]))
            taus.append(r_opt.x[0] / r_opt.x[1])
            ls_results.append(r_opt)
        except Exception as e:
            pass
            
    if len(ls_results) != 3:
        log_failure(uid_dis, 'fit', 'multi_start_fail', 'Not all starts converged')
        return None
        
    tau_spread = max(taus) - min(taus)
    multi_start_flag = 1 if tau_spread > 1.0 else 0
    
    res = ls_results[0]
    mCp, hA = res.x
    rmse = np.sqrt(np.mean(res.fun**2))
    tau_B = taus[0]
    tau_B_ctrl = taus[1]
    tau_AB_diff_s = abs(tau_B - tau_B_ctrl)
    
    nfev = res.nfev
    ls_status = res.status
    optimality = res.optimality
    
    # Boundary check
    tol = 1e-4
    boundary_flag = 1 if (mCp <= 5+tol or mCp >= 150-tol or hA <= 0.005+tol or hA >= 0.5-tol) else 0
    
    # Quality gate
    quality_pass = 1 if (boundary_flag == 0 and rmse <= 2.0 and len(t_vals) >= 30) else 0
    
    # Jacobian for correlation
    J = res.jac
    try:
        cov = np.linalg.inv(J.T.dot(J))
        corr_mCp_hA = cov[0,1] / np.sqrt(cov[0,0] * cov[1,1])
    except:
        corr_mCp_hA = np.nan
        
    quality_pass = 1 if (quality_pass == 1 and multi_start_flag == 0) else 0

    return {
        'uid': uid_dis, 'cycle_index': cycle_index, 'mCp_JK': mCp, 'hA_WK': hA,
        'tau_B_s': tau_B, 'rmse_K': rmse, 'n_points': len(t_vals),
        'n_excluded_transient': 0, # computed outside
        'n_out_of_grid': n_out_of_grid, 'n_negQ': n_negQ,
        'corr_mCp_hA': corr_mCp_hA, 'boundary_flag': boundary_flag,
        'quality_pass': quality_pass, 'tau_AB_diff_s': tau_AB_diff_s,
        'nfev': nfev, 'ls_status': ls_status, 'optimality': optimality,
        'tau_spread_3start_s': tau_spread, 'multi_start_flag': multi_start_flag
    }

print("Loaded definitions.")

from tqdm import tqdm

for i, (uid_dis, uid_chg, d_df, c_df) in enumerate(tqdm(valid_pairs, desc="Base Fit")):
    res = fit_cycle(uid_dis, d_df, c_df, i+1)
    if res:
        res['n_excluded_transient'] = len([x for x in excluded_points if x['uid'] == uid_dis])
        fit_results.append(res)

df_fit = pd.DataFrame(fit_results)
df_pass = df_fit[df_fit['quality_pass'] == 1].copy()

n_boundary = len(df_fit[df_fit['boundary_flag'] == 1])
n_rmse_fail = len(df_fit[(df_fit['boundary_flag'] == 0) & (df_fit['rmse_K'] > 2.0)])
n_too_few = len(df_fit[(df_fit['boundary_flag'] == 0) & (df_fit['rmse_K'] <= 2.0) & (df_fit['n_points'] < 30)])
n_exception = len(valid_pairs) - len(df_fit)
n_quality_pass = len(df_pass)
n_multi_start_fail = len(df_fit[df_fit['multi_start_flag'] == 1])

# Some failures log to w2b_failures.csv without appending to fit_results (e.g. no_overlap, empty_grid)
n_curve_or_data_fail = len([x for x in failures if x['error_type'] in ['no_overlap', 'empty_grid']])

accounting = [
    {'category': 'n_discharge_total', 'count': len(b0005[b0005['type']=='discharge'])},
    {'category': 'n_unpaired', 'count': n_unpaired},
    {'category': 'n_paired', 'count': n_paired},
    {'category': 'n_curve_or_data_fail', 'count': n_curve_or_data_fail},
    {'category': 'n_quality_pass', 'count': n_quality_pass},
    {'category': 'n_boundary', 'count': n_boundary},
    {'category': 'n_rmse_fail', 'count': n_rmse_fail},
    {'category': 'n_too_few_points', 'count': n_too_few},
    {'category': 'n_multi_start_fail', 'count': n_multi_start_fail},
    {'category': 'n_exception_during_fit', 'count': n_exception - n_curve_or_data_fail}
]
pd.DataFrame(accounting).to_csv(os.path.join(out_dir, 'w2b_accounting.csv'), index=False)

if len(excluded_points) == 0:
    pd.DataFrame(columns=['uid', 'row_index', 'Time', 'Current_measured', 'reason']).to_csv(os.path.join(out_dir, 'w2b_excluded_points.csv'), index=False)
else:
    pd.DataFrame(excluded_points).to_csv(os.path.join(out_dir, 'w2b_excluded_points.csv'), index=False)

if len(failures) == 0:
    pd.DataFrame(columns=['uid', 'stage', 'error_type', 'message']).to_csv(os.path.join(out_dir, 'w2b_failures.csv'), index=False)
else:
    pd.DataFrame(failures).to_csv(os.path.join(out_dir, 'w2b_failures.csv'), index=False)

df_fit.to_csv(os.path.join(out_dir, 'w2b_fit_by_cycle.csv'), index=False)
pd.DataFrame(cycle_grids).to_csv(os.path.join(out_dir, 'w2b_eocp_grid.csv'), index=False)

# Bootstrap for sigma_boot
tau_B_median = df_pass['tau_B_s'].median()
mCp_median = df_pass['mCp_JK'].median()
hA_median = df_pass['hA_WK'].median()

np.random.seed(20260825)
B = 2000
boot_medians = []
for _ in range(B):
    sample = df_pass['tau_B_s'].sample(n=len(df_pass), replace=True)
    boot_medians.append(sample.median())
sigma_boot = np.std(boot_medians)

# dUdT sensitivity (§6)
dudt_sens = []
for dudt_val in [-0.5, -0.2, 0.0, 0.2, 0.5]:
    # convert mV/K to V/K
    dudt_vk = dudt_val / 1000.0
    # Refit all valid pairs
    d_res = []
    for i, (uid_dis, uid_chg, d_df, c_df) in enumerate(valid_pairs):
        r = fit_cycle(uid_dis, d_df, c_df, i+1, dudt=dudt_vk, Tamb=T_AMB_BASE)
        if r and r['quality_pass'] == 1:
            d_res.append(r['tau_B_s'])
    
    tau_med = np.median(d_res) if d_res else np.nan
    dudt_sens.append({
        'dUdT_mVK': dudt_val,
        'tau_B_median_s': tau_med,
        'n_pass': len(d_res)
    })
df_dudt = pd.DataFrame(dudt_sens)
df_dudt.to_csv(os.path.join(out_dir, 'w2b_dudt_sensitivity.csv'), index=False)
sigma_dudt = (df_dudt['tau_B_median_s'].max() - df_dudt['tau_B_median_s'].min()) / 2.0

# Sigma budget (§6)
sigma_b = np.sqrt(SIGMA_B_OCV**2 + sigma_boot**2 + sigma_dudt**2)
budget = [
    {'component': 'sigma_OCV', 'value_s': SIGMA_B_OCV, 'origin': 'registered', 'formula': '91.67 s (registered)'},
    {'component': 'sigma_boot', 'value_s': sigma_boot, 'origin': 'computed', 'formula': 'bootstrap median std (B=2000)'},
    {'component': 'sigma_dUdT', 'value_s': sigma_dudt, 'origin': 'computed', 'formula': '(max-min)/2 of 5 runs'},
    {'component': 'sigma_B_total', 'value_s': sigma_b, 'origin': 'computed', 'formula': 'sqrt(sum of squares)'}
]
pd.DataFrame(budget).to_csv(os.path.join(out_dir, 'w2b_sigma_budget.csv'), index=False)

# Compat check (§5)
pass_2 = abs(TAU_A - tau_B_median) <= 2 * np.sqrt(SIGMA_A**2 + sigma_b**2)
pd.DataFrame([{
    'lhs_s': abs(TAU_A - tau_B_median),
    'rhs_s': 2 * np.sqrt(SIGMA_A**2 + sigma_b**2),
    'sigma_B_s': sigma_b,
    'pass': pass_2
}]).to_csv(os.path.join(out_dir, 'w2b_compat_check.csv'), index=False)

# T_amb scan (§7)
tamb_scan = []
for t_val in [23.0, 23.5, 24.0, 24.5, 25.0, 26.0]:
    t_res = []
    t_mcp = []
    t_ha = []
    for i, (uid_dis, uid_chg, d_df, c_df) in enumerate(valid_pairs):
        r = fit_cycle(uid_dis, d_df, c_df, i+1, dudt=0.0, Tamb=t_val)
        if r and r['quality_pass'] == 1:
            t_res.append(r['tau_B_s'])
            t_mcp.append(r['mCp_JK'])
            t_ha.append(r['hA_WK'])
    tamb_scan.append({
        'T_amb_C': t_val,
        'tau_B_median_s': np.median(t_res) if t_res else np.nan,
        'hA_median_WK': np.median(t_ha) if t_ha else np.nan,
        'mCp_median_JK': np.median(t_mcp) if t_mcp else np.nan,
        'n_pass': len(t_res)
    })
df_tamb = pd.DataFrame(tamb_scan)
df_tamb.to_csv(os.path.join(out_dir, 'w2b_tamb_scan.csv'), index=False)


# X4 Objective Function Plot
if len(df_pass) > 0:
    mid_uid = df_pass.iloc[(df_pass['rmse_K'] - df_pass['rmse_K'].median()).abs().argsort()[:1]]['uid'].values[0]
    r_opt = [x for x in fit_results if x['uid'] == mid_uid][0]
    
    # re-extract data
    c_idx = next(i for i, v in enumerate(valid_pairs) if v[0] == mid_uid)
    d_df, c_df = valid_pairs[c_idx][2], valid_pairs[c_idx][3]
    Q_d_min, Q_d_max = d_df['Q_fromfull'].min(), d_df['Q_fromfull'].max()
    Q_c_min, Q_c_max = c_df['Q_fromfull'].min(), c_df['Q_fromfull'].max()
    overlap_lo = max(Q_d_min, Q_c_min)
    overlap_hi = min(Q_d_max, Q_c_max)
    grid = np.arange(np.ceil(overlap_lo/0.05)*0.05, overlap_hi, 0.05)
    d_cc_s = d_df.sort_values('Q_fromfull')
    c_cc_s = c_df.sort_values('Q_fromfull')
    V_dis_interp = np.interp(grid, d_cc_s['Q_fromfull'], d_cc_s['Voltage_measured'])
    V_chg_interp = np.interp(grid, c_cc_s['Q_fromfull'], c_cc_s['Voltage_measured'])
    I_chg_interp = np.interp(grid, c_cc_s['Q_fromfull'], c_cc_s['Current_measured'].abs())
    I_dis = np.abs(d_cc_s['Current_measured'].median())
    E_weighted_grid = (I_dis * V_chg_interp + I_chg_interp * V_dis_interp) / (I_chg_interp + I_dis)
    
    cc_end_idx = d_df.index[d_df['Voltage_measured'] < 2.8].min()
    fit_df = d_df if pd.isna(cc_end_idx) else d_df.loc[:cc_end_idx]
    t_vals = fit_df['Time'].values
    T_meas = fit_df['Temperature_measured'].values + 273.15
    I_vals = np.abs(fit_df['Current_measured'].values)
    V_vals = fit_df['Voltage_measured'].values
    Q_vals = fit_df['Q_fromfull'].values
    in_grid = (Q_vals >= grid[0]) & (Q_vals <= grid[-1])
    t_vals = t_vals[in_grid]
    T_meas = T_meas[in_grid]
    I_vals = I_vals[in_grid]
    V_vals = V_vals[in_grid]
    Q_vals = Q_vals[in_grid]
    E_ocp_vals = np.interp(Q_vals, grid, E_weighted_grid)
    T0_K = T_meas[0]
    
    def calc_rmse(m, h):
        T_model = solve_thermal_model(t_vals, T0_K, I_vals, V_vals, E_ocp_vals, m, h, 0.0, T_AMB_BASE)
        return np.sqrt(np.mean((T_model - T_meas)**2))
        
    mCp_opt = r_opt['mCp_JK']
    hA_opt = r_opt['hA_WK']
    mCp_arr = np.linspace(max(5, mCp_opt - 20), min(150, mCp_opt + 20), 10)
    hA_arr = np.linspace(max(0.005, hA_opt - 0.02), min(0.5, hA_opt + 0.02), 10)
    
    rmse_grid = np.zeros((10, 10))
    for i, h in enumerate(hA_arr):
        for j, m in enumerate(mCp_arr):
            rmse_grid[i, j] = calc_rmse(m, h)
            
    # Check if minimum is not at opt
    min_idx = np.unravel_index(np.argmin(rmse_grid), rmse_grid.shape)
    mCp_min_grid = mCp_arr[min_idx[1]]
    hA_min_grid = hA_arr[min_idx[0]]
    dm = abs(mCp_min_grid - mCp_opt) / (mCp_arr[1] - mCp_arr[0])
    dh = abs(hA_min_grid - hA_opt) / (hA_arr[1] - hA_arr[0])
    print(f"X4 Check: grid min at ({mCp_min_grid:.2f}, {hA_min_grid:.4f}), opt at ({mCp_opt:.2f}, {hA_opt:.4f})")
    if dm > 1.5 or dh > 1.5:
        print(f"WARNING: X4 Check failed! Least squares minimum is far from grid minimum.")
    
    # 1D slice
    mCp_fine = np.linspace(max(5, mCp_opt - 20), min(150, mCp_opt + 20), 50)
    rmse_slice = [calc_rmse(m, hA_opt) for m in mCp_fine]
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    cp = plt.contourf(mCp_arr, hA_arr, rmse_grid, levels=20, cmap='viridis_r')
    plt.colorbar(cp, label='RMSE (K)')
    plt.plot(mCp_opt, hA_opt, 'ro', label='least_squares opt')
    plt.plot(40, 0.05, 'wX', label='Start A')
    plt.xlabel('mCp (J/K)')
    plt.ylabel('hA (W/K)')
    plt.legend()
    plt.title(f'RMSE Landscape (uid={mid_uid})')
    
    plt.subplot(1, 2, 2)
    plt.plot(mCp_fine, rmse_slice, 'b-', label=f'RMSE at hA={hA_opt:.4f}')
    plt.plot(mCp_opt, calc_rmse(mCp_opt, hA_opt), 'ro', label='opt')
    plt.plot(40, calc_rmse(40, hA_opt), 'wX', markeredgecolor='k', label='Start A mCp')
    plt.xlabel('mCp (J/K)')
    plt.ylabel('RMSE (K)')
    plt.legend()
    plt.title('1D Slice along mCp')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'fig_W2b_0_objective.png'))
    plt.close()

# Make plots

# 1. Fits
plt.figure(figsize=(10, 8))
if len(df_pass) > 0:
    best_uid = df_pass.loc[df_pass['rmse_K'].idxmin(), 'uid']
    worst_uid = df_pass.loc[df_pass['rmse_K'].idxmax(), 'uid']
    mid_uid = df_pass.iloc[(df_pass['rmse_K'] - df_pass['rmse_K'].median()).abs().argsort()[:1]]['uid'].values[0]
    
    uids_to_plot = [(best_uid, 'Best'), (mid_uid, 'Median'), (worst_uid, 'Worst')]
    for i, (uid, label) in enumerate(uids_to_plot):
        r = [x for x in fit_results if x['uid'] == uid][0]
        c_idx = next(i for i, v in enumerate(valid_pairs) if v[0] == uid)
        d_df, c_df = valid_pairs[c_idx][2], valid_pairs[c_idx][3]
        
        Q_d_min, Q_d_max = d_df['Q_fromfull'].min(), d_df['Q_fromfull'].max()
        Q_c_min, Q_c_max = c_df['Q_fromfull'].min(), c_df['Q_fromfull'].max()
        overlap_lo = max(Q_d_min, Q_c_min)
        overlap_hi = min(Q_d_max, Q_c_max)
        grid = np.arange(np.ceil(overlap_lo/0.05)*0.05, overlap_hi, 0.05)
        
        d_cc_s = d_df.sort_values('Q_fromfull')
        c_cc_s = c_df.sort_values('Q_fromfull')
        V_dis_interp = np.interp(grid, d_cc_s['Q_fromfull'], d_cc_s['Voltage_measured'])
        V_chg_interp = np.interp(grid, c_cc_s['Q_fromfull'], c_cc_s['Voltage_measured'])
        I_chg_interp = np.interp(grid, c_cc_s['Q_fromfull'], c_cc_s['Current_measured'].abs())
        I_dis = np.abs(d_cc_s['Current_measured'].median())
        E_weighted_grid = (I_dis * V_chg_interp + I_chg_interp * V_dis_interp) / (I_chg_interp + I_dis)
        
        cc_end_idx = d_df.index[d_df['Voltage_measured'] < 2.8].min()
        fit_df = d_df if pd.isna(cc_end_idx) else d_df.loc[:cc_end_idx]
        t_vals = fit_df['Time'].values
        T_meas = fit_df['Temperature_measured'].values + 273.15
        I_vals = np.abs(fit_df['Current_measured'].values)
        V_vals = fit_df['Voltage_measured'].values
        Q_vals = fit_df['Q_fromfull'].values
        
        in_grid = (Q_vals >= grid[0]) & (Q_vals <= grid[-1])
        t_vals = t_vals[in_grid]
        T_meas = T_meas[in_grid]
        I_vals = I_vals[in_grid]
        V_vals = V_vals[in_grid]
        Q_vals = Q_vals[in_grid]
        E_ocp_vals = np.interp(Q_vals, grid, E_weighted_grid)
        
        T_model = solve_thermal_model(t_vals, T_meas[0], I_vals, V_vals, E_ocp_vals, r['mCp_JK'], r['hA_WK'], 0.0, T_AMB_BASE)
        
        ax1 = plt.subplot(3, 2, i*2 + 1)
        ax1.plot(t_vals, T_meas - 273.15, 'k.', label='Measured')
        ax1.plot(t_vals, T_model - 273.15, 'r-', label=f'Model ({label}, RMSE={r["rmse_K"]:.2f}K)')
        ax1.legend(loc='best')
        ax1.set_ylabel('T (°C)')
        
        ax2 = plt.subplot(3, 2, i*2 + 2)
        Q_irr = I_vals * (E_ocp_vals - V_vals)
        ax2.plot(t_vals, Q_irr, 'g-', label='Q_irr')
        ax2.axhline(0, color='gray', linestyle='--')
        ax2.text(t_vals[0], min(Q_irr), "Q_rev=0 (handled by envelope)", color='gray', fontsize=8)
        ax2.legend(loc='best')
        ax2.set_ylabel('Heat Gen (W)')
        
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'fig_W2b_1_fits.png'))
plt.close()

# 2. Tau by cycle
plt.figure()
plt.scatter(df_pass['cycle_index'], df_pass['tau_B_s'], c='blue', label='tau_B')
plt.axhline(TAU_A, color='r', linestyle='-', label='tau_A = 841s')
tol = 2 * np.sqrt(SIGMA_A**2 + sigma_b**2)
plt.axhspan(TAU_A - tol, TAU_A + tol, color='r', alpha=0.2, label='Tolerance Window')
plt.xlabel('Cycle Index')
plt.ylabel('Tau_B (s)')
plt.legend()
plt.title('Tau_B by Cycle')
plt.savefig(os.path.join(out_dir, 'fig_W2b_2_tau_by_cycle.png'))
plt.close()

# 3. hA vs mCp
plt.figure()
sc = plt.scatter(df_pass['mCp_JK'], df_pass['hA_WK'], c=df_pass['cycle_index'], cmap='viridis')
plt.colorbar(sc, label='Cycle Index')
plt.xlabel('mCp (J/K)')
plt.ylabel('hA (W/K)')
plt.title('hA vs mCp Parameter Distribution')
plt.savefig(os.path.join(out_dir, 'fig_W2b_3_hA_vs_mCp.png'))
plt.close()

# 4. Tamb scan
plt.figure()
plt.plot(df_tamb['T_amb_C'], df_tamb['tau_B_median_s'], 'mo-')
plt.xlabel('T_amb (°C)')
plt.ylabel('Tau_B Median (s)')
plt.title('Sensitivity to T_amb')
plt.savefig(os.path.join(out_dir, 'fig_W2b_4_tamb_scan.png'))
plt.close()

# 5. dudt sens
plt.figure()
plt.plot(df_dudt['dUdT_mVK'], df_dudt['tau_B_median_s'], 'co-')
plt.xlabel('dUdT (mV/K)')
plt.ylabel('Tau_B Median (s)')
plt.title('Sensitivity to dU/dT')
plt.savefig(os.path.join(out_dir, 'fig_W2b_5_dudt_sens.png'))
plt.close()

print("W2b_process Python execution finished.")

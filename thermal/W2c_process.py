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
parser.add_argument('--w2b-dir', default='output')
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


def solve_thermal_model(t, T0, I_t, V_t, E_ocp_t, mCp, hA, dudt_prof, Tamb):
    T = np.zeros_like(t)
    T[0] = T0
    def dTdt(t_val, T_val):
        i = np.interp(t_val, t, I_t)
        v = np.interp(t_val, t, V_t)
        e = np.interp(t_val, t, E_ocp_t)
        du = np.interp(t_val, t, dudt_prof) if isinstance(dudt_prof, np.ndarray) else dudt_prof
        Q_gen = i * (e - v) + i * T_val * du
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


def solve_2node_model(t, T0, I_t, V_t, E_ocp_t, mCp, hA, tau_lag, Tamb):
    T_lump, T_meas = np.zeros_like(t), np.zeros_like(t)
    T_lump[0], T_meas[0] = T0, T0
    def derivs(t_val, Tl, Tm):
        i = np.interp(t_val, t, I_t)
        v = np.interp(t_val, t, V_t)
        e = np.interp(t_val, t, E_ocp_t)
        Q_gen = i * (e - v)
        dTl = (Q_gen - hA * (Tl - (Tamb + 273.15))) / mCp
        dTm = (Tl - Tm) / tau_lag
        return dTl, dTm

    for i in range(len(t) - 1):
        t_curr, Tl_curr, Tm_curr, t_next = t[i], T_lump[i], T_meas[i], t[i+1]
        dt_full = t_next - t_curr
        if dt_full <= 0:
            T_lump[i+1], T_meas[i+1] = Tl_curr, Tm_curr
            continue
        n_steps = max(1, int(np.ceil(dt_full / 5.0)))
        dt = dt_full / n_steps
        Tl_temp, Tm_temp = Tl_curr, Tm_curr
        for _ in range(n_steps):
            t_step = t_curr + _ * dt
            k1l, k1m = derivs(t_step, Tl_temp, Tm_temp)
            k2l, k2m = derivs(t_step + dt/2, Tl_temp + dt*k1l/2, Tm_temp + dt*k1m/2)
            k3l, k3m = derivs(t_step + dt/2, Tl_temp + dt*k2l/2, Tm_temp + dt*k2m/2)
            k4l, k4m = derivs(t_step + dt, Tl_temp + dt*k3l, Tm_temp + dt*k3m)
            Tl_temp += (dt/6.0) * (k1l + 2*k2l + 2*k3l + k4l)
            Tm_temp += (dt/6.0) * (k1m + 2*k2m + 2*k3m + k4m)
        T_lump[i+1], T_meas[i+1] = Tl_temp, Tm_temp
    return T_meas

def extract_vectors(d_df, c_df):
    overlap_lo = max(d_df['Q_fromfull'].min(), c_df['Q_fromfull'].min())
    overlap_hi = min(d_df['Q_fromfull'].max(), c_df['Q_fromfull'].max())
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
    return t_vals[in_grid], T_meas[in_grid], I_vals[in_grid], V_vals[in_grid], Q_vals[in_grid], np.interp(Q_vals[in_grid], grid, E_weighted_grid), d_df['Q_fromfull'].max()

def run_multi_start(starts, residuals, bounds):
    ls_res = []
    taus = []
    for s0 in starts:
        try:
            r = least_squares(residuals, s0, bounds=bounds)
            taus.append(r.x[0] / r.x[1])
            ls_res.append(r)
        except: pass
    if len(ls_res) != len(starts): return None, None, None
    tau_spread = max(taus) - min(taus)
    multi_flag = 1 if tau_spread > 1.0 else 0
    return ls_res[0], tau_spread, multi_flag

df_w2b = pd.read_csv(os.path.join(args.w2b_dir, 'w2b_fit_by_cycle.csv'))
pass_uids = df_w2b[df_w2b['quality_pass'] == 1]['uid'].tolist()
valid_pairs_pass = [p for p in valid_pairs if p[0] in pass_uids]

d1_results, d2_results, d3_results, d4_results = [], [], [], []
starts_w2 = [[40.0, 0.05], [20.0, 0.02], [60.0, 0.08]]
starts_d3 = [[40.0, 0.05, 100], [80.0, 0.038, 10]]
T_AMB = 24.0

print(f"Running D1-D4 on {len(valid_pairs_pass)} passed pairs...")

for i, (uid_dis, uid_chg, d_df, c_df) in enumerate(tqdm(valid_pairs_pass)):
    try:
        t_all, Tm_all, I_all, V_all, Q_all, E_all, Qd_max = extract_vectors(d_df, c_df)
    except: continue
    
    d4_results.append({
        'uid': uid_dis,
        'T_rest_end_C': d_df['Temperature_measured'].iloc[-5:].median(),
        'dev_from_24_K': d_df['Temperature_measured'].iloc[-5:].median() - 24.0,
        'rest_duration_s': d_df['Time'].iloc[-1] - d_df['Time'].iloc[d_df[d_df['Current_measured'] < -0.1].index.max()]
    })

    # D1
    for wname, mask in [('early', t_all - t_all[0] <= 600), ('late', t_all - t_all[0] >= 1000)]:
        t_w, Tm_w, I_w, V_w, Q_w, E_w = t_all[mask], Tm_all[mask], I_all[mask], V_all[mask], Q_all[mask], E_all[mask]
        if len(t_w) < 5: continue
        T0_w = Tm_w[0]
        def res_d1(p):
            Tm_model = solve_thermal_model(t_w, T0_w, I_w, V_w, E_w, p[0], p[1], 0.0, T_AMB)
            return Tm_model - Tm_w
        r, t_sp, m_f = run_multi_start(starts_w2, res_d1, ([5, 0.005], [150, 0.5]))
        if r:
            mCp, hA = r.x
            rmse = np.sqrt(np.mean(r.fun**2))
            try:
                J = r.jac
                cov = np.linalg.inv(J.T.dot(J))
                corr = cov[0,1] / np.sqrt(cov[0,0] * cov[1,1])
            except: corr = np.nan
            b_flag = 1 if (mCp<=5.01 or mCp>=149.9 or hA<=0.0051 or hA>=0.499) else 0
            d1_results.append({'uid': uid_dis, 'window': wname, 'mCp_JK': mCp, 'hA_WK': hA, 'tau_s': mCp/hA, 'rmse_K': rmse, 'n_points': len(t_w), 'corr': corr, 'boundary_flag': b_flag, 'multi_start_flag': m_f})

    # D2
    soc = 1 - Q_all / Qd_max
    profiles = {
        'P0': np.zeros_like(soc),
        'P1': 1.0 * (soc - 0.5) / 1000.0,
        'P2': -1.0 * (soc - 0.5) / 1000.0,
        'P3': np.where(soc > 0.5, -0.3, 0.3) / 1000.0,
        'P4': np.where(soc > 0.5, 0.3, -0.3) / 1000.0
    }
    knee_mask = (t_all - t_all[0] >= 300) & (t_all - t_all[0] <= 900)
    for pname, prof in profiles.items():
        def res_d2(p):
            # Since solve_thermal_model now accepts dudt_prof (array) instead of scalar, we modify our usage or redefine solve_thermal_model 
            # WAIT! The original solve_thermal_model in W2b accepts a scalar `dudt`! 
            # I need to redefine it here to take array dudt_prof.
            pass

    for pname, prof in profiles.items():
        def res_d2(p):
            return solve_thermal_model(t_all, Tm_all[0], I_all, V_all, E_all, p[0], p[1], prof, T_AMB) - Tm_all
        r, _, m_f = run_multi_start(starts_w2, res_d2, ([5, 0.005], [150, 0.5]))
        if r and m_f == 0:
            mCp, hA = r.x
            rmse = np.sqrt(np.mean(r.fun**2))
            rmse_knee = np.sqrt(np.mean(r.fun[knee_mask]**2)) if knee_mask.sum() > 0 else np.nan
            d2_results.append({'uid': uid_dis, 'profile': pname, 'mCp_JK': mCp, 'hA_WK': hA, 'tau_s': mCp/hA, 'rmse_K': rmse, 'rmse_knee_K': rmse_knee})

    # D3
    def res_d3(p):
        return solve_2node_model(t_all, Tm_all[0], I_all, V_all, E_all, p[0], p[1], p[2], T_AMB) - Tm_all
    try:
        r1 = least_squares(res_d3, starts_d3[0], bounds=([5, 0.005, 1], [150, 0.5, 600]))
        r2 = least_squares(res_d3, starts_d3[1], bounds=([5, 0.005, 1], [150, 0.5, 600]))
        r = r1 if r1.cost <= r2.cost else r2
        mCp, hA, tau_lag = r.x
        rmse = np.sqrt(np.mean(r.fun**2))
        try:
            cov = np.linalg.inv(r.jac.T.dot(r.jac))
            corr = np.max(np.abs(np.corrcoef(cov) - np.eye(3)))
        except: corr = np.nan
        b_flag = 1 if (mCp<=5.01 or mCp>=149.9 or hA<=0.0051 or hA>=0.499 or tau_lag<=1.01 or tau_lag>=599.9) else 0
        d3_results.append({'uid': uid_dis, 'mCp_JK': mCp, 'hA_WK': hA, 'tau_lag_s': tau_lag, 'tau_lump_s': mCp/hA, 'rmse_K': rmse, 'corr_max_abs': corr, 'boundary_flag': b_flag, 'n_points': len(t_all)})
    except: pass

out_dir_c = args.out_dir
os.makedirs(out_dir_c, exist_ok=True)

df_d1 = pd.DataFrame(d1_results)
df_full = df_w2b[df_w2b['uid'].isin(df_d1['uid'].unique())].copy()
df_full['window'] = 'full'
df_full['tau_s'] = df_full['tau_B_s']
df_full['corr'] = df_full['corr_mCp_hA']
df_d1 = pd.concat([df_d1, df_full[['uid', 'window', 'mCp_JK', 'hA_WK', 'tau_s', 'rmse_K', 'n_points', 'corr', 'boundary_flag', 'multi_start_flag']]])
df_d1.to_csv(os.path.join(out_dir_c, 'w2c_window_split.csv'), index=False)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
for i, w in enumerate(['early', 'late', 'full']):
    d = df_d1[df_d1['window'] == w]['mCp_JK']
    plt.scatter([i]*len(d), d, alpha=0.5, label=w)
plt.xticks([0, 1, 2], ['early', 'late', 'full'])
plt.ylabel('mCp (J/K)')
plt.subplot(1, 2, 2)
for i, w in enumerate(['early', 'late', 'full']):
    d = df_d1[df_d1['window'] == w]['tau_s']
    plt.scatter([i]*len(d), d, alpha=0.5, label=w)
plt.xticks([0, 1, 2], ['early', 'late', 'full'])
plt.ylabel('tau (s)')
plt.savefig(os.path.join(out_dir_c, 'fig_W2c_1_window_split.png'))
plt.close()

df_d2 = pd.DataFrame(d2_results)
agg_d2 = df_d2.groupby('profile').agg(
    tau_B_median_s=('tau_s', 'median'),
    mCp_median_JK=('mCp_JK', 'median'),
    hA_median_WK=('hA_WK', 'median'),
    rmse_median_K=('rmse_K', 'median'),
    rmse_knee_K=('rmse_knee_K', 'median'),
    n_pass=('uid', 'count')
).reset_index()
agg_d2.to_csv(os.path.join(out_dir_c, 'w2c_shape_family.csv'), index=False)

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.bar(agg_d2['profile'], agg_d2['tau_B_median_s'])
plt.ylabel('Tau Median (s)')
plt.subplot(2, 1, 2)
plt.bar(agg_d2['profile'], agg_d2['rmse_knee_K'])
plt.ylabel('RMSE Knee (K)')
plt.savefig(os.path.join(out_dir_c, 'fig_W2c_2_shape_family.png'))
plt.close()

df_d3 = pd.DataFrame(d3_results)
df_d3.to_csv(os.path.join(out_dir_c, 'w2c_twonode_fit.csv'), index=False)

if len(df_d3) > 0:
    df_d3_sort = df_d3.sort_values('rmse_K')
    uids = [df_d3_sort.iloc[0]['uid'], df_d3_sort.iloc[len(df_d3)//2]['uid'], df_d3_sort.iloc[-1]['uid']]
    plt.figure(figsize=(12, 10))
    for i, uid in enumerate(uids):
        ax = plt.subplot(3, 1, i+1)
        r_d3 = df_d3[df_d3['uid'] == uid].iloc[0]
        r_d1 = df_w2b[df_w2b['uid'] == uid].iloc[0]
        
        c_idx = next(idx for idx, v in enumerate(valid_pairs_pass) if v[0] == uid)
        d_df, c_df = valid_pairs_pass[c_idx][2], valid_pairs_pass[c_idx][3]
        t_all, Tm_all, I_all, V_all, Q_all, E_all, Qd_max = extract_vectors(d_df, c_df)
        
        Tm_d1 = solve_thermal_model(t_all, Tm_all[0], I_all, V_all, E_all, r_d1['mCp_JK'], r_d1['hA_WK'], 0.0, T_AMB)
        Tm_d3 = solve_2node_model(t_all, Tm_all[0], I_all, V_all, E_all, r_d3['mCp_JK'], r_d3['hA_WK'], r_d3['tau_lag_s'], T_AMB)
        
        ax.plot(t_all, Tm_all - 273.15, 'k.', label='Measured')
        ax.plot(t_all, Tm_d1 - 273.15, 'r-', label=f'1-Node (RMSE {r_d1["rmse_K"]:.2f})')
        ax.plot(t_all, Tm_d3 - 273.15, 'b--', label=f'2-Node (RMSE {r_d3["rmse_K"]:.2f}, lag={r_d3["tau_lag_s"]:.0f}s)')
        ax.set_ylabel('T (°C)')
        ax.legend()
        if i == 0: ax.set_title('Two-Node Lag Diagnositc (M2)')
        
        axins = ax.inset_axes([0.6, 0.1, 0.35, 0.4])
        axins.plot(t_all, Tm_all - 273.15, 'k.')
        axins.plot(t_all, Tm_d1 - 273.15, 'r-')
        axins.plot(t_all, Tm_d3 - 273.15, 'b--')
        axins.set_xlim(300, 900)
        axins.set_ylim((Tm_all - 273.15)[(t_all >= 300) & (t_all <= 900)].min() - 0.5, (Tm_all - 273.15)[(t_all >= 300) & (t_all <= 900)].max() + 0.5)
        ax.indicate_inset_zoom(axins, edgecolor="black")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir_c, 'fig_W2c_3_twonode.png'))
    plt.close()

df_d4 = pd.DataFrame(d4_results)
df_d4.to_csv(os.path.join(out_dir_c, 'w2c_rest_ambient.csv'), index=False)
plt.figure()
plt.hist(df_d4['T_rest_end_C'], bins=20, edgecolor='black')
plt.axvline(24.0, color='r', linestyle='--', label='24.0 °C')
plt.xlabel('Rest End Temperature (°C)')
plt.ylabel('Count')
plt.title('Ambient Drift Check (M3)')
plt.legend()
plt.savefig(os.path.join(out_dir_c, 'fig_W2c_4_rest_ambient.png'))
plt.close()

print("Done with W2c processing.")

# 繪圖已移至 make_w5_fig.py（W6b）；本檔只負責跑模擬與輸出 CSV。
import pybamm
import pandas as pd
import numpy as np

# 關閉 pybamm warnings 避免洗版
pybamm.set_logging_level("ERROR")

param_sets = ["Marquis2019", "Chen2020", "Ecker2015", "Prada2013"]

# 實驗設定
experiment_steps = [
    (
        "Discharge at 7.5C for 300 seconds",
        "Charge at 7.5C for 300 seconds",
    )
] * 2 + [
    (
        "Discharge at 7.5C for 300 seconds",
        "Rest for 600 seconds",
    )
]
experiment = pybamm.Experiment(experiment_steps)
model = pybamm.lithium_ion.SPM(options={"thermal": "lumped"})

results = []
failures = []
curves_data = []

for idx, p_name in enumerate(param_sets):
    print(f"Processing {p_name}...")
    
    # 提取參數集基本資訊
    try:
        param = pybamm.ParameterValues(p_name)
        nom_cap = param.get("Nominal cell capacity [A.h]")
        t_init = param.get("Initial temperature [K]")
        t_amb = param.get("Ambient temperature [K]")
    except Exception as e:
        failures.append({
            "param_set": p_name,
            "stage": "load_parameters",
            "error_type": type(e).__name__,
            "message": str(e).split('\n')[0]
        })
        continue
    
    # 執行模擬
    sol = None
    try:
        sim = pybamm.Simulation(model, experiment=experiment, parameter_values=param)
        sol = sim.solve()
    except Exception as e:
        failures.append({
            "param_set": p_name,
            "stage": "simulation",
            "error_type": type(e).__name__,
            "message": str(e).split('\n')[0]
        })
    
    # 整理結果
    if sol is not None:
        term = sol.termination
        solved = term == "final time"
        term_reason = "completed" if solved else term
        
        t_sec = sol["Time [s]"].entries
        t_comp = t_sec[-1]
        
        T_arr = sol["Volume-averaged cell temperature [K]"].entries
        t_init_sim = T_arr[0]
        dT = T_arr - t_init_sim
        dt_max = np.max(dT)
        t_max = np.max(T_arr)
        
        try:
            n_cycles = len(sol.cycles)
        except:
            n_cycles = 0
            
        results.append({
            "param_set": p_name,
            "nominal_capacity_Ah": nom_cap,
            "T_init_K": t_init,
            "T_amb_K": t_amb,
            "solved": solved,
            "termination_reason": term_reason,
            "t_completed_s": t_comp,
            "dT_max_K": dt_max,
            "T_max_K": t_max,
            "n_full_cycles_completed": n_cycles
        })
        
        # Output curve data
        for t, temp_rise in zip(t_sec, dT):
            curves_data.append({
                "param_set": p_name,
                "t_s": t,
                "dT_K": temp_rise
            })
            
    else:
        results.append({
            "param_set": p_name,
            "nominal_capacity_Ah": nom_cap,
            "T_init_K": t_init,
            "T_amb_K": t_amb,
            "solved": False,
            "termination_reason": "Failed to run",
            "t_completed_s": np.nan,
            "dT_max_K": np.nan,
            "T_max_K": np.nan,
            "n_full_cycles_completed": np.nan
        })

# X1 可重現性檢查
marquis = next((r for r in results if r["param_set"] == "Marquis2019"), None)
if marquis is not None:
    # 基準 2.004 K：Figure_PyBaMM_Temp.png 像素量測（146 px = 0.25 K，峰在 row 99）
    diff = abs(marquis["dT_max_K"] - 2.004)
    if diff > 0.1:
        print(f"!!! WARNING: Marquis2019 dT_max_K is {marquis['dT_max_K']:.4f} K. Expected ~2.004 K. Difference: {diff:.4f} K")
        print(f"PyBaMM version: {pybamm.__version__}")

# 輸出 CSV
df_res = pd.DataFrame(results)
df_res.to_csv("output/w5_param_sets.csv", index=False)

df_fail = pd.DataFrame(failures, columns=["param_set", "stage", "error_type", "message"])
df_fail.to_csv("output/w5_failures.csv", index=False)

df_curves = pd.DataFrame(curves_data)
df_curves.to_csv("output/w5_curves.csv", index=False)

import sys, platform, scipy, json
env_info = {
    "pybamm.__version__": pybamm.__version__,
    "numpy.__version__": np.__version__,
    "scipy.__version__": scipy.__version__,
    "python_version": platform.python_version(),
    "timestamp": pd.Timestamp.now().isoformat()
}
with open("output/environment.json", "w") as f:
    json.dump(env_info, f, indent=4)


print("Simulation and CSV output completed.")

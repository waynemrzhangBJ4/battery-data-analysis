import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# 設定中文字型
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Microsoft JhengHei', 'PingFang SC', 'SimHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

curves_df = pd.read_csv('output/w5_curves.csv')
res_df = pd.read_csv('output/w5_param_sets.csv')

param_sets = ["Marquis2019", "Chen2020", "Ecker2015", "Prada2013"]

fig, ax = plt.subplots(figsize=(10, 6))
colors = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange']
axins = ax.inset_axes([0.4, 0.4, 0.5, 0.4])

for idx, p_name in enumerate(param_sets):
    if p_name == "Prada2013":
        continue
    p_curve = curves_df[curves_df['param_set'] == p_name]
    if len(p_curve) == 0:
        continue
        
    t_sec = p_curve['t_s'].values
    dT = p_curve['dT_K'].values
    
    ax.plot(t_sec, dT, color=colors[idx], linewidth=2, label=p_name)
    if p_name in ["Marquis2019", "Ecker2015"]:
        axins.plot(t_sec, dT, color=colors[idx], linewidth=2)
        
    solved = res_df.loc[res_df['param_set'] == p_name, 'solved'].values[0]
    t_comp = res_df.loc[res_df['param_set'] == p_name, 't_completed_s'].values[0]
    print(f"{p_name}: {t_comp} -> {t_comp:.0f} s")
    
    if not solved:
        ax.plot(t_sec[-1], dT[-1], marker='X', color=colors[idx], markersize=10, label=f"{p_name} (提前終止)")
        ax.text(t_sec[-1] + 40, dT[-1], f"{t_comp:.0f} s", color=colors[idx], va='center', ha='left', fontweight='bold')
    else:
        ax.text(t_sec[-1], dT[-1] + 1.5, f"{t_comp:.0f} s", color=colors[idx], va='bottom', ha='right', fontweight='bold')

# 調整圖表
ax.set_ylim(0, 45) # 寫死
ax.set_xlabel('Time (s)')
ax.set_ylabel('Temperature Rise ΔT (K)')
ax.grid(True)
ax.set_title("同一組 7.5C 負載，四組參數集", pad=32)
t_chen = res_df.loc[res_df['param_set'] == 'Chen2020', 't_completed_s'].values[0]
t_ecker = res_df.loc[res_df['param_set'] == 'Ecker2015', 't_completed_s'].values[0]
t_marquis = res_df.loc[res_df['param_set'] == 'Marquis2019', 't_completed_s'].values[0]
sub_title = ax.text(0.5, 1.02, f"三組的終止時間不同：{t_chen:.0f} / {t_ecker:.0f} / {t_marquis:.0f} s", transform=ax.transAxes, ha='center', va='bottom', fontsize=11)

# W6e measurements
fig.canvas.draw()
main_title = ax.title
ax_e = ax.get_window_extent()
sub_e = sub_title.get_window_extent()
main_e = main_title.get_window_extent()

ax_y1 = ax_e.y1
sub_y0, sub_y1 = sub_e.y0, sub_e.y1
main_y0, main_y1 = main_e.y0, main_e.y1

gap1 = sub_y0 - ax_y1
status1 = "OK" if gap1 > 0 else "壓線"
gap2 = main_y0 - sub_y1
status2 = "OK" if gap2 > 0 else "重疊"

main_cx = (main_e.x0 + main_e.x1)/2
sub_cx = (sub_e.x0 + sub_e.x1)/2
cx_diff = main_cx - sub_cx

print("--- 標題區檢查 ---")
print(f"繪圖區上框線 y1        : {ax_y1:.2f}")
print(f"副標   y0~y1           : {sub_y0:.2f}~{sub_y1:.2f}")
print(f"主標題 y0~y1           : {main_y0:.2f}~{main_y1:.2f}")
print(f"副標下緣 - 上框線       : {gap1:.2f} px   判定: {status1}")
print(f"主標題下緣 - 副標上緣   : {gap2:.2f} px   判定: {status2}")
print(f"主標題水平中心 - 副標水平中心 : {cx_diff:.2f} px")

# 調整內嵌小圖
axins.set_ylim(0, 3) # 寫死
axins.set_title("放大：ΔT 0–3 K（Marquis2019 & Ecker2015）", fontsize=10)
axins.grid(True)

# indicate_inset_zoom removed per W6c

# 確保不重複顯示 label (因為有些有提前終止標記，有些沒有)
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
final_handles = list(by_label.values())
final_labels = list(by_label.keys())

# Add Prada2013 empty entry
final_handles.append(Line2D([], [], color='gray', linestyle='none', marker=''))
final_labels.append('Prada2013：參數不足，未能求解')

ax.legend(final_handles, final_labels, loc='upper left', bbox_to_anchor=(1.05, 1))

fig.tight_layout()
plt.savefig('output/fig_W5_1_param_sets.png', dpi=300, bbox_inches='tight')

print("make_w5_fig done.")

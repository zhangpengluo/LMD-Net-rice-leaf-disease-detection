import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO
import warnings

warnings.filterwarnings('ignore')


# ============================================================
# 全局样式
# ============================================================
def set_plot_style():
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.family': 'Times New Roman',
        'font.size': 30,
        'axes.titlesize': 36,
        'axes.labelsize': 32,
        'xtick.labelsize': 30,
        'ytick.labelsize': 30,
        'legend.fontsize': 15,
        'lines.linewidth': 3.5,          # ← 全局加粗
        'figure.dpi': 300,
        'savefig.dpi': 600,
        'savefig.bbox': 'tight',
        'axes.linewidth': 2.0,           # ← 轴线加粗
        'axes.edgecolor': '#333333',
        'axes.grid': False,
        'axes.facecolor': 'white',
        'figure.facecolor': 'white',
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'xtick.major.size': 6,
        'ytick.major.size': 6,
        'xtick.minor.size': 3,
        'ytick.minor.size': 3,
        'axes.axisbelow': False,
    })


COLORS = ['#2563EB', '#DC2626', '#059669', '#D97706', '#7C3AED',
          '#DB2777', '#0891B2', '#4F46E5', '#EA580C', '#16A34A',
          '#64748B', '#BE185D', '#0D9488', '#B45309']

LEGEND_COMPACT = dict(
    fontsize=15, frameon=True, fancybox=True, framealpha=0.9,
    edgecolor='#ccc', borderpad=0.3, labelspacing=0.25,
    handlelength=1.2, handletextpad=0.4, borderaxespad=0.3, columnspacing=0.8,
)


def ema_smooth(values, weight=0.6):
    s = np.zeros_like(values, dtype=float)
    s[0] = values[0]
    for i in range(1, len(values)):
        s[i] = weight * s[i - 1] + (1 - weight) * values[i]
    return s


def format_axes(ax):
    ax.tick_params(axis='both', which='both',
                   top=False, right=False,
                   bottom=True, left=True,
                   direction='out', labelsize=30)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_facecolor('white')
    ax.grid(False)

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    xticks = ax.get_xticks()
    yticks = ax.get_yticks()

    xticks_ok = xticks[(xticks > 0) & (xticks <= xlim[1])]
    yticks_ok = yticks[(yticks >= 0) & (yticks <= ylim[1])]

    if len(xticks_ok) > 0:
        ax.set_xticks(xticks_ok)
    if len(yticks_ok) > 0:
        ax.set_yticks(yticks_ok)
        labels = ['0.0' if v == 0 else f'{v:.1f}' for v in yticks_ok]
        ax.set_yticklabels(labels)

    x_range = xlim[1] - xlim[0] if xlim[1] != xlim[0] else 1.0
    y_range = ylim[1] - ylim[0] if ylim[1] != ylim[0] else 1.0
    pad = 0.06

    ax.set_xlim(xlim[0], xlim[1] + x_range * pad)
    ax.set_ylim(ylim[0], ylim[1] + y_range * pad)

    ax.spines['bottom'].set_bounds(xlim[0], xlim[1])
    ax.spines['left'].set_bounds(ylim[0], ylim[1])
    ax.spines['left'].set_zorder(100)
    ax.spines['bottom'].set_zorder(100)

    arrow_kw = dict(arrowstyle='->', color='#333333', lw=2.0,
                    shrinkA=0, shrinkB=0, mutation_scale=15)
    ax.annotate('', xy=(xlim[1] + x_range * pad, ylim[0]),
                xytext=(xlim[1], ylim[0]),
                arrowprops=arrow_kw, annotation_clip=False, zorder=100)
    ax.annotate('', xy=(xlim[0], ylim[1] + y_range * pad),
                xytext=(xlim[0], ylim[1]),
                arrowprops=arrow_kw, annotation_clip=False, zorder=100)


# ============================================================
# 从模型验证提取曲线数据
# ============================================================
def extract_curves(model_path, data_yaml):
    print('🔍 加载模型并运行验证...')
    model = YOLO(model_path)
    results = model.val(data=data_yaml, plots=False, verbose=True)

    names = model.names
    if isinstance(names, dict):
        class_names = [names[i] for i in sorted(names.keys())]
    else:
        class_names = list(names)
    nc = len(class_names)
    print(f'📋 类别数: {nc}, 类别: {class_names}')

    metric = results.box
    curves = {}
    px = np.array(metric.px)
    print(f'  ✓ px shape: {px.shape}')

    p_curve = r_curve = f1_curve = None

    if hasattr(metric, 'p_curve') and metric.p_curve is not None:
        p_curve = np.array(metric.p_curve)
        print(f'  ✓ p_curve shape: {p_curve.shape}')
    if hasattr(metric, 'r_curve') and metric.r_curve is not None:
        r_curve = np.array(metric.r_curve)
        print(f'  ✓ r_curve shape: {r_curve.shape}')
    if hasattr(metric, 'f1_curve') and metric.f1_curve is not None:
        f1_curve = np.array(metric.f1_curve)
        print(f'  ✓ f1_curve shape: {f1_curve.shape}')

    if p_curve is None and hasattr(metric, 'curves_results'):
        cr = metric.curves_results
        print(f'  📦 curves_results: type={type(cr)}, len={len(cr)}')
        for i, item in enumerate(cr):
            if isinstance(item, (list, tuple)):
                print(f'    [{i}] 元组, 长度={len(item)}')
                for j, sub in enumerate(item):
                    sub_arr = np.array(sub)
                    print(f'      [{i}][{j}] shape={sub_arr.shape}')
            else:
                arr = np.array(item)
                print(f'    [{i}] shape={arr.shape}')
        if len(cr) >= 4:
            if isinstance(cr[0], (list, tuple)) and len(cr[0]) == 2:
                px = np.array(cr[0][0]); p_curve = np.array(cr[0][1])
            else:
                p_curve = np.array(cr[0])
            if isinstance(cr[1], (list, tuple)) and len(cr[1]) == 2:
                r_curve = np.array(cr[1][1])
            else:
                r_curve = np.array(cr[1])
            if isinstance(cr[3], (list, tuple)) and len(cr[3]) == 2:
                f1_curve = np.array(cr[3][1])
            else:
                f1_curve = np.array(cr[3])
            print(f'  ✓ 从 curves_results 提取成功')

    if p_curve is None and hasattr(metric, 'prec_values'):
        pv = np.array(metric.prec_values)
        print(f'  📦 prec_values shape: {pv.shape}')
        if pv.ndim == 2:
            p_curve = pv

    def fix_shape(arr, nc_, n_conf):
        if arr is None:
            return None
        arr = np.array(arr)
        if arr.ndim == 1:
            return arr.reshape(1, -1)
        if arr.shape[0] == n_conf and arr.shape[1] == nc_:
            return arr.T
        return arr

    n_conf = len(px)
    p_curve  = fix_shape(p_curve,  nc, n_conf)
    r_curve  = fix_shape(r_curve,  nc, n_conf)
    f1_curve = fix_shape(f1_curve, nc, n_conf)

    if p_curve is not None:
        print(f'  ✓ p_curve 最终 shape: {p_curve.shape}')
    if r_curve is not None:
        print(f'  ✓ r_curve 最终 shape: {r_curve.shape}')
    if f1_curve is not None:
        print(f'  ✓ f1_curve 最终 shape: {f1_curve.shape}')

    ap50 = np.array(metric.ap50) if hasattr(metric, 'ap50') else None

    if p_curve is not None:
        curves['P'] = {
            'x': px, 'y_per_class': p_curve, 'y_all': p_curve.mean(axis=0),
            'class_names': class_names,
            'x_label': 'Confidence', 'y_label': 'Precision',
            'title': 'Precision-Confidence Curve',
        }
    if r_curve is not None:
        curves['R'] = {
            'x': px, 'y_per_class': r_curve, 'y_all': r_curve.mean(axis=0),
            'class_names': class_names,
            'x_label': 'Confidence', 'y_label': 'Recall',
            'title': 'Recall-Confidence Curve',
        }
    if f1_curve is not None:
        curves['F1'] = {
            'x': px, 'y_per_class': f1_curve, 'y_all': f1_curve.mean(axis=0),
            'class_names': class_names,
            'x_label': 'Confidence', 'y_label': 'F1',
            'title': 'F1-Confidence Curve',
        }
    if p_curve is not None and r_curve is not None:
        map50 = ap50.mean() if ap50 is not None else 0
        curves['PR'] = {
            'x': r_curve, 'y_per_class': p_curve,
            'x_all': r_curve.mean(axis=0), 'y_all': p_curve.mean(axis=0),
            'class_names': class_names,
            'x_label': 'Recall', 'y_label': 'Precision',
            'title': 'Precision-Recall Curve',
            'is_pr': True, 'ap50': ap50, 'map50': map50,
        }

    print(f'\n📊 成功提取 {len(curves)} 条曲线: {list(curves.keys())}')
    return curves


# ============================================================
# ★★ 核心：4 行 × 3 列 合成大图
#   Row 1: train/box_loss  | train/cls_loss  | train/dfl_loss
#   Row 2: val/box_loss    | val/cls_loss    | val/dfl_loss
#   Row 3: Precision       | Recall          | mAP@0.5
#   Row 4: mAP@0.5:0.95   | P-Confidence    | R-Confidence
# ============================================================
def plot_combined_all(csv_path, curves_data, save_path, smooth=0.6, show_best=False):
    set_plot_style()
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    # ---------- 10 个训练 / 验证指标 ----------
    train_configs = [
        ('train/box_loss',       'Box Loss (Train)',  'Loss'),
        ('train/cls_loss',       'Cls Loss (Train)',  'Loss'),
        ('train/dfl_loss',       'DFL Loss (Train)',  'Loss'),
        ('val/box_loss',         'Box Loss (Val)',    'Loss'),
        ('val/cls_loss',         'Cls Loss (Val)',    'Loss'),
        ('val/dfl_loss',         'DFL Loss (Val)',    'Loss'),
        ('metrics/precision(B)', 'Precision',         'Precision'),
        ('metrics/recall(B)',    'Recall',            'Recall'),
        ('metrics/mAP50(B)',     'mAP@0.5',           'mAP'),
        ('metrics/mAP50-95(B)',  'mAP@0.5:0.95',      'mAP'),
    ]
    valid_train = [(c, t, y) for c, t, y in train_configs if c in df.columns]

    N_ROWS, N_COLS = 4, 3
    fig, axes = plt.subplots(N_ROWS, N_COLS, figsize=(9 * N_COLS, 7 * N_ROWS))
    axes_flat = axes.flatten()
    epochs = np.arange(1, len(df) + 1)
    used = set()

    # ========== 位置 0‒9：训练指标 ==========
    for idx, (col, title, ylabel) in enumerate(valid_train[:10]):
        ax = axes_flat[idx]
        used.add(idx)

        vals = df[col].values
        sm = ema_smooth(vals, smooth)
        color = COLORS[idx % len(COLORS)]

        ax.plot(epochs, vals, color=color, alpha=0.20, linewidth=2.0, zorder=1)
        ax.plot(epochs, sm,   color=color, linewidth=3.5, zorder=2)

        # ---- 删除/关闭 Best 标注 ----
        if show_best:
            is_loss = 'loss' in col.lower()
            best_i = np.argmin(sm) if is_loss else np.argmax(sm)
            best_v = sm[best_i]
            ax.scatter(epochs[best_i], best_v, color=color, s=120,
                       zorder=5, edgecolors='white', linewidths=2.5)
            offset_y = 15 if is_loss else -20
            ax.annotate(f'Best: {best_v:.4f}',
                        xy=(epochs[best_i], best_v),
                        xytext=(15, offset_y), textcoords='offset points',
                        fontsize=20, color=color, fontweight='bold',
                        arrowprops=dict(arrowstyle='->', color=color, lw=2.0),
                        bbox=dict(boxstyle='round,pad=0.3', fc='white',
                                  ec=color, alpha=0.8),
                        zorder=6)

        ax.set_title(title, fontweight='bold', pad=10)
        ax.set_xlabel('Epoch', fontweight='bold')
        ax.set_ylabel(ylabel, fontweight='bold')
        ax.set_xlim([1, len(epochs)])
        format_axes(ax)

    # ========== 位置 10、11：P-Confidence / R-Confidence ==========
    for ci, key in enumerate(['P', 'R']):
        pos = 10 + ci
        ax = axes_flat[pos]

        if key not in curves_data:
            continue
        used.add(pos)

        cd = curves_data[key]
        y_cls = cd['y_per_class']
        nc = y_cls.shape[0]

        for i in range(nc):
            color = COLORS[i % len(COLORS)]
            ax.plot(cd['x'], y_cls[i], color=color, lw=2.5,
                    alpha=0.7, label=cd['class_names'][i], zorder=1)

        x_a, y_a = cd['x'], cd['y_all']
        ax.fill_between(x_a, y_a, alpha=0.08, color='#1a1a1a', zorder=1)

        # ---- 均值线：不要包含 best 数值的 legend 文本 ----
        ax.plot(x_a, y_a, color='#1a1a1a', lw=4.5, label='All', zorder=10)

        # ---- 删除/关闭 Best 点与坐标标注 ----
        if show_best:
            best_i = np.argmax(y_a)
            best_v, best_x = y_a[best_i], x_a[best_i]
            ax.scatter(best_x, best_v, color='#DC2626', s=120,
                       zorder=15, edgecolors='white', linewidths=2.5)
            ax.annotate(f'({best_x:.2f}, {best_v:.3f})',
                        xy=(best_x, best_v),
                        xytext=(20, -15), textcoords='offset points',
                        fontsize=20, fontweight='bold', color='#DC2626',
                        arrowprops=dict(arrowstyle='->', color='#DC2626', lw=2.0),
                        bbox=dict(boxstyle='round,pad=0.3', fc='white',
                                  ec='#DC2626', alpha=0.85),
                        zorder=16)

        ax.set_xlabel(cd['x_label'], fontweight='bold')
        ax.set_ylabel(cd['y_label'], fontweight='bold')
        ax.set_title(cd['title'], fontweight='bold', pad=10)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])
        format_axes(ax)

        ncol_leg = 2 if nc > 5 else 1
        leg = ax.legend(loc='best', ncol=ncol_leg, **LEGEND_COMPACT)
        leg.get_frame().set_linewidth(0.8)
        leg.set_zorder(20)

    # ========== 隐藏未使用的子图 ==========
    for i in range(N_ROWS * N_COLS):
        if i not in used:
            axes_flat[i].set_visible(False)

    # ========== 行标题（左侧竖排标注） ==========
    row_labels = ['Training Loss',
                  'Validation Loss',
                  'Metrics',
                  'mAP & Confidence Curves']
    for r, label in enumerate(row_labels):
        fig.text(0.005, 1 - (r + 0.5) / N_ROWS,
                 label, va='center', ha='left',
                 fontsize=28, fontweight='bold', color='#555555',
                 rotation=90)

    fig.suptitle('Training Results  ·  Precision-Confidence  ·  Recall-Confidence',
                 fontsize=40, fontweight='bold', y=1.008)
    fig.patch.set_facecolor('white')
    plt.tight_layout(w_pad=3, h_pad=4, rect=[0.025, 0, 1, 0.98])
    fig.savefig(save_path, dpi=600, bbox_inches='tight', facecolor='white')
    print(f'✅ 4×3 合成大图已保存: {save_path}')
    plt.close()


# ============================================================
# 主程序入口
# ============================================================
if __name__ == '__main__':

    # ============================================
    # ⚙️ 配置路径
    # ============================================
    TRAIN_DIR  = r'F:\rice leaf disease\yolov13-main\aaa\train8'
    MODEL_PATH = r'F:\rice leaf disease\yolov13-main\aaa\train8\weights\best.pt'
    DATA_YAML  = r'F:\rice leaf disease\datasets_new\dataset.yaml'
    OUTPUT_DIR = r'F:\rice leaf disease\trian_img'
    # ============================================

    train_dir = Path(TRAIN_DIR)
    save_dir  = Path(OUTPUT_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)

    print('=' * 60)
    print('🎨 YOLOv11 美化绘图工具 — 4×3 合成大图')
    print(f'📂 输出目录: {save_dir}')
    print('=' * 60)

    csv_path = train_dir / 'results.csv'

    if not csv_path.exists():
        print(f'❌ 未找到 results.csv: {csv_path}')
    else:
        print('\n📊 [Step 1] 提取曲线数据...')
        curves = extract_curves(MODEL_PATH, DATA_YAML)

        if curves:
            print('\n📊 [Step 2] 绘制 4×3 合成大图...')
            plot_combined_all(
                csv_path=str(csv_path),
                curves_data=curves,
                save_path=save_dir / 'combined_4x3_beautiful.png',
                smooth=0.6,
            )
        else:
            print('❌ 未能提取 P/R 曲线数据')

    print('\n' + '=' * 60)
    print('🎉 全部完成！')
    for f in sorted(save_dir.glob('*_beautiful.png')):
        print(f'   📄 {f.name}')
    print('=' * 60)
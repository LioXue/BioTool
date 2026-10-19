import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]   #字体
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False              #显示负号


def linear_fit_plot(
    x,
    y,
    out_path,
    title="Linear Fit",
    xlabel="x",
    ylabel="y",
    x_unit="",
    y_unit="",
):

    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()

    if x.size != y.size:
        raise ValueError("x 与 y 的长度必须相同")
    if x.size < 2:
        raise ValueError("至少需要 2 个数据点才能进行线性拟合")

    slope, intercept = np.polyfit(x, y, 1)

    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    def _with_unit(name, unit):
        name = (name or "").strip()
        unit = (unit or "").strip()
        if unit:
            return f"{name} ({unit})" if name else f"({unit})"
        return name if name else " "

    xlabel_full = _with_unit(xlabel, x_unit)
    ylabel_full = _with_unit(ylabel, y_unit)

    #画布尺寸：figsize=(宽, 高)，单位英寸；dpi 影响清晰度
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)

    ax.scatter(
        x, y,
        s=40,                   # 点大小
        color="purple",         # 点颜色
        alpha=0.85,             # 透明度 0~1
        edgecolors="white",     # 点的描边颜色
        linewidths=0.6,         # 点的描边粗细
        label="Data points",    # 图例文字
    )

    # ★ 拟合直线样式：color=线颜色，lw=线宽
    x_line = np.linspace(x.min(), x.max(), 200)
    ax.plot(
        x_line, slope * x_line + intercept,
        color="black",        # 线颜色（红色）
        lw=1,                   # 线宽
        label="Fitted line",
    )

    sign = "+" if intercept >= 0 else "-"
    eq = f"y = {slope:.4f}x {sign} {abs(intercept):.4f}"
    ax.text(
        0.05, 0.90,                          # 文字位置（相对坐标 0~1）
        f"{eq}\n$R^2$ = {r2:.4f}",
        transform=ax.transAxes,
        va="top", ha="left",
        fontsize=12,                         #方程文字字号
        # fontweight="bold",                 #加粗取消注释
    )

    ax.set_xlabel(xlabel_full, fontsize=12)  # 横轴名称字号
    ax.set_ylabel(ylabel_full, fontsize=12)  # 纵轴名称字号
    ax.set_title(title, fontsize=14)         # 图标题字号

    ax.tick_params(axis="both", labelsize=10)#坐标刻度数字的字号

    #ax.legend(
     #   loc="lower right",
     #   fontsize=10,   # 图例字号
     #   frameon=True,  # 显示图例外框
     #   facecolor="white",  # 框内填充色
     #   edgecolor="gray",  # 框边颜色
     #   framealpha=0.9,  # 框的透明度
     #   borderpad=0.5,  # 框内边距
     #   )

    ax.grid(alpha=0.3)                       #网格线

    fig.tight_layout()

    fig.savefig(out_path, format="jpg", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"拟合方程 : {eq}")
    print(f"斜率 k   : {slope:.6f}")
    print(f"截距 b   : {intercept:.6f}")
    print(f"R^2      : {r2:.6f}")
    print(f"横轴     : {xlabel_full}")
    print(f"纵轴     : {ylabel_full}")
    print(f"图标题   : {title}")
    print(f"图像已保存: {out_path}")

    return {
        "slope": slope,
        "intercept": intercept,
        "r2": r2,
        "equation": eq,
        "image_path": out_path,
    }

if __name__ == "__main__":
    result = linear_fit_plot(
        [0,0.5,1,1.5,2,2.5,3,4],
        [0,0.09,0.218,0.254,0.349,0.417,0.517,0.696],
        out_path=r"C:\Users\Xuely\Desktop\生物科学\大学生物实验\生物学实验2\蛋白质含量定量测定uv法.jpg",
        title="UV法标准曲线",
        xlabel="mass of protein",
        ylabel="A 280nm",
        x_unit="mg",
        y_unit="",
    )
    print(result)
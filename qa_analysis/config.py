import matplotlib.pyplot as plt
import seaborn as sns

ANCHOR_DATE = '2025-02-17'  # 自然周锚点
DEFAULT_REPLACEMENT_HOURS = 168.0  # 无穷值默认替换值(一周)

def setup_plotting():
    # 字体配置（Mac）
    plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Hiragino Sans GB', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    sns.set_style("whitegrid")
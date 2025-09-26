
# ----------------------------------------------------------------------------------------------------
# 合并矢量图
# ----------------------------------------------------------------------------------------------------

# import fitz  # PyMuPDF
# import matplotlib.pyplot as plt
#
#
# def merge_pdfs_preserve_vector(input_paths, output_path):
#     merged_pdf = fitz.open()
#     for path in input_paths:
#         pdf = fitz.open(path)
#         merged_pdf.insert_pdf(pdf)  # 插入完整PDF（保留所有矢量内容）
#         pdf.close()
#     merged_pdf.save(output_path)
#     merged_pdf.close()
#
#
# # 使用示例
# merge_pdfs_preserve_vector(["file1.pdf", "file2.pdf"], "merged_vector.pdf")

# ----------------------------------------------------------------------------------------------------
# 核心可视化结果
# ----------------------------------------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

Q_learning = np.load('Q-learning_step_counter_times.npy')
SARSA = np.load('SARSA_step_counter_times.npy')

plt.subplots(1, 1, figsize=(6, 6))  # 绘制矢量图，Q表格 和时间步数图

# 绘制子图1，创建表格
plt.subplot(1, 1, 1)
plt.plot(Q_learning, 'g-', color='blue', label='Q-Learning')
plt.plot(SARSA, 'g-', color='red', label='SARSA')
std_dev1 = np.array([np.std(Q_learning[:i+1]) for i in range(len(Q_learning))])  # 标准差
std_dev2 = np.array([np.std(SARSA[:i+1]) for i in range(len(SARSA))])  # 标准差
plt.fill_between(np.arange(len(Q_learning)), Q_learning - std_dev1, Q_learning + std_dev1, color='blue', alpha=0.1, label=None)
plt.fill_between(np.arange(len(SARSA)), SARSA - std_dev2, SARSA + std_dev2, color='red', alpha=0.1, label=None)
plt.xlabel('episode num')
plt.ylabel("steps")
plt.title('Steps in each episode', fontsize=14, loc="center", pad=20)
plt.legend()  # 显示图例
plt.grid(True, alpha=0.3)  # 添加半透明网格
plt.tight_layout()  # 自动优化间隔布局
plt.savefig("Q-learning_SARSA.pdf", format='pdf', bbox_inches='tight')
plt.show()
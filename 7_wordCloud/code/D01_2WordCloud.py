"""
基于给定的CSV文件，给定词语和频率
"""
import wordcloud
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib

matplotlib.use('Agg')  # 使用非交互式后端来保存图像

def generate_wordcloud_from_csv(csv_path, num):
    """
    从指定的CSV文件生成词云图。

    :param csv_path: CSV文件的路径。
    """
    # 确保使用绝对路径读取CSV文件，以增强脚本的可移植性
    csv_path = Path(csv_path).resolve()

    # 读取CSV数据
    df = pd.read_csv(csv_path, encoding='gbk')

    # 初始化两个列表用来存储单词和词频
    words = []
    frequencies = []


    # 准备词云数据
    # 假设csv文件中，单词在偶数列，词频在奇数列
    for i in range(0, len(df.columns), 2):
        words.append(df.iloc[:, i].fillna('').tolist())
        frequencies.append(df.iloc[:, i + 1].fillna(0).tolist())

    # 打印结果
    for i in range(len(words)):
        print(f"Words: {words[i]}")
        print(f"Frequencies: {frequencies[i]}")
        # 生成词云
        wordcloud = WordCloud(width=800, height=400, background_color='white', font_path='/mnt/c/Windows/Fonts/宋体.ttf').generate_from_frequencies(
            dict(zip(words[i], frequencies[i])))

        # 显示词云图
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')  # 隐藏坐标轴
        plt.tight_layout(pad=0)  # 优化布局，去除多余的边距
        # plt.show()
        plt.savefig(Path(f'数据{i+1}第{num}词云图.jpg').resolve(), bbox_inches='tight', pad_inches=0, dpi=300)
        plt.close()


# 确定CSV文件的位置
csv_file_path = Path('词云图数据.csv').resolve()

# 调用函数生成词云
for i in range(1, 10):
    generate_wordcloud_from_csv(csv_file_path, i)

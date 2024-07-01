"""
using the data we provide to pretrain the model
"""
from snownlp import sentiment

def train_self_model():
    # 训练结束后会在输出目录得到一个 .marshal.3 的文件:
    pos = "./pos.txt"
    neg = "./neg.txt"
    sentiment.train(neg, pos)
    sentiment.save("sentiment.marshal")

def get_site_pkg_path():
    # 获取Site - Packages路径
    import site
    # Add snownlp/sentiment
    return site.getsitepackages()[0]
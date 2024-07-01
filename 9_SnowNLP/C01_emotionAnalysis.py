"""
test
"""
from snownlp import SnowNLP


def sentiment_analysis(text):
    # 使用SnowNLP对中文文本进行情感分析
    s = SnowNLP(text)
    # SnowNLP的sentiments方法返回情感倾向分数，越接近1表明情感越积极，越接近0表明情感越消极
    sentiment_score = s.sentiments
    return sentiment_score

text = "角色塑造太单调，毫无震撼力！"

score = sentiment_analysis(text)
print(f"情感分数: {score}")

if score > 0.5:
    print("该语句是积极的。")
else:
    print("该语句是消极的。")
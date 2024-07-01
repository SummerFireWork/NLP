
# 一次生成
num_topics = 5 # 主题数
n_topics_word = 20 # 展示前N个主题词
passes = 10 # 最大迭代次数
processed_language = "english"  #or english/chinese/all

# 多次迭代
start_value = 5 # 开始迭代数
n_max_topics = 50 # 最大主题数量
length = 1 # 每次迭代跳过的主题数量
max_iter = 10 # 最大迭代次数

a = range(start_value, n_max_topics, length)
print(range(start_value, n_max_topics, length))
print(list(a))
print(len(list(a)))
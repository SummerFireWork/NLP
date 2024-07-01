"""
第一步：本程序目的是对中文文件进行分词，仅需要修改输入、输出文件即可。
"""
# 导入jieba模块及其扩展
import jieba
import jieba.analyse
import jieba.posseg as pseg
import codecs, sys
import re
import os


def get_stop_dict(file):
    content = open(file,encoding="utf-8")
    word_list = []
    for c in content:
        c = re.sub('\n|\r','',c)
        word_list.append(c)
    return word_list


# 打开输入与输出文件
f = codecs.open("01theFirstOne.txt", "r", "utf-8")
target = codecs.open("01theFirstOne.target.txt", "w", "utf-8")
print("open files")

os.chdir("D:/pythonProject5")
stop_words = get_stop_dict('stopwords/baidu_stopwords.txt')+get_stop_dict('stopwords/hit_stopwords.txt')+get_stop_dict('stopwords/cn_stopwords.txt')+get_stop_dict('stopwords/scu_stopwords.txt')+get_stop_dict('stopwords/own_stopwords.txt')
print(stop_words)


# 对文件中的每一行进行分词，并将结果写入输出文件
line_num = 1  # 行号初始化
line = f.readline()  # 读取文件第一行
while line:
    print("line:", line_num)  # 打印当前处理的行号
    # line_seg = " ".join(jieba.cut(line))  # 对当前行进行分词，旧版本，未增加停用词
    line_seg = " ".join(word for word in jieba.cut(line) if word not in stop_words and len(word)>1)
    target.writelines(line_seg)  # 将分词结果写入输出文件
    line_num = line_num + 1  # 行号自增
    line = f.readline()  # 读取下一行
f.close()  # 关闭输入文件
target.close()  # 关闭输出文件

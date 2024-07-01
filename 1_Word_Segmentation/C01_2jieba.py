"""
使用jieba给txt进行分词， 输出字表、词表cs.txt,vocab.txt
"""
import jieba
import jieba.analyse

# tips：dict有去重效果，得到没有重复值的内容
# dict = {}
# dict['a'] = 1
# dict['a'] = 2
# print(dict)

vocab = []
cs = []
text = ''
with open('D:\\pythonProject5\\1_Word_Segmentation\\4中国优秀桥梁.txt', 'r', encoding='utf-8') as file:
    for line in file:
        # 将所有空格、换行去掉
        line = line.strip()
        # 将每个句子添加到变量text中，用来做分析
        text +=line
        # 输出每句话
        # print(line)
        # 进行分词，结果是一个列表
        result = jieba.cut(line)
        # 取出每一个词，得到词表，此表格式是列表
        for word in result:
            vocab.append(word)
        # 如果要取字表，更加简单
        for c in line:
            cs.append(c)


# 字表去重
cs = list(set(cs))
# 词表去重
vocab = list(set(vocab))

# 输出字表
with open('C01-2cs.txt', 'w', encoding='utf-8') as csf:
    for c in cs:
        csf.write(c+'\n')
# 输出词表
with open('C01-2vocab.txt', 'w', encoding='utf-8') as vc:
    for w in vocab:
        vc.write(w+'\n')

print("="*200)
# 找出词频最高的10个词语
TextRank = jieba.analyse.textrank(text, topK=10, withWeight=False)
print(TextRank)
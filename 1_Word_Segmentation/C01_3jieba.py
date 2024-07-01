"""
使用jieba给txt进行分词， 输出字表、词表
"""
import jieba.analyse
import jieba.posseg as pseg

# 使用pseg分词，得到带有词性的结果
vocab = pseg.cut('将文本内容划分为词库和字库')
print(list(vocab))
print("="*200)

#############################################################################
# 1. 得到分词（带有词性）表
text = ''
with open('D:\\pythonProject5\\1_Word_Segmentation\\4中国优秀桥梁.txt', 'r', encoding='utf-8') as file:
    with open('C01-3seg1.txt', 'w', encoding='utf-8') as segfile:
        # get every sentence line
        for line in file:
            # delect or skip the space
            line = line.strip()
            if line!='':
                # run the tokenizer for every line, but get the rsult content "pair"
                for p in pseg.cut(line):
                    # 1 type: write the result of tokenizer and use the space separate them
                    # segfile.write(str(p)+' ')
                    # 2 type: writh the tokenizer with textrank
                    p = list(p)
                    segfile.write(p[0]+'/'+p[1]+' ')
                segfile.write('\n')
        print('分词结果保存成功')
        segfile.close()

########################################################################

# 2. 得到词语、词性词频分析结果
# text 用来保存文本，用来给之后的关键词提取
text = ''
lines = []
with open('D:\\pythonProject5\\1_Word_Segmentation\\4中国优秀桥梁.txt', 'r', encoding='utf-8') as file:
    text = file.read()
    # get every sentence line

for line in text.strip('\n'):
    # delect or skip the space
    line=line.strip()
    if line!='':
        lines.append(list(pseg.cut(line)))
print('分词词性标注完成')

# 将上述结果保存
with open('C01-3seg2.txt', 'w', encoding='utf-8') as segfile:
    for line in lines:
        for p in line:
            p = list(p)
            segfile.write(p[0]+'/'+p[1]+' ')
        segfile.write('\n')
    print('分词结果保存成功')
    segfile.close()
##################################################################################################
# 根据上述结果，进行结果计算和排序
wc = {}
with open('C01-4wc.txt', 'w', encoding='utf-8') as wcFile:
    for l in lines:
        for p in l:
            p = list(p)
            # 如果wc中，没有p[0]这个词，就设置为1，如果有就加1
            if wc.get(p[0]):
                wc[p[0]] += 1
            else:
                wc[p[0]] =1
    for key in wc.keys():
        wcFile.write(key + ' '+ str(wc[key]) + '\n')
    print('词频结果保存成功')
#######################################################################################################################
# part of sentence词性
posc = {}
with open('C01-5posc.txt', 'w', encoding='utf-8') as poscFile:
    for l in lines:
        for p in l:
            p = list(p)
            # 如果posc中，没有p[1]这个词性，就设置为1，如果有就加1
            if posc.get(p[1]):
                posc[p[1]] += 1
            else:
                posc[p[1]] =1
    for key in posc.keys():
        poscFile.write(key + ' '+ str(posc[key]) + '\n')
    print('词性频率结果保存成功')

######################################################################################################
################################################################################################################
# 对于完整的多行文本
# 2.1使用TextRank进行关键词提取
a = jieba.analyse.textrank(text, topK=10, withWeight=False)
print(a)

##################################################################################################################

# 2.2统计次数最多的十个成语
# 以词性统计的结果为基础, 得到成语的词频字典
cys = {}
with open('D:\\pythonProject5\\1_Word_Segmentation\\C01-3seg1.txt', 'r', encoding='utf-8') as rmrbfile:
    for l in rmrbfile:
        # 取得每行的结果，以空格作为间隔
        for pstr in l.split(' '):
            # 去除每行的空格，换行符，得到每个词语和词性，列表。
            p = pstr.split('/')
            # 得到成语
            if len(p) > 1 and p[1] == 'i':
                if cys.get(p[0]):
                    cys[p[0]] += 1
                else:
                    cys[p[0]] = 1
    print(cys)

# 进行排序
cyList = []
for cy in cys.keys():
    cyList.append([cy, cys[cy]])
# 打印所有列表
print(sorted(cyList, key=lambda  item:item[1], reverse=True))
# 打印前十个
cyTop10 = sorted(cyList, key=lambda item:item[1], reverse=True)[:10]
for cy in cyTop10:
    print(cy)

############################################################################################################
# 2.3词性种类最多的10个词
# 以词性统计的结果为基础, 得到词性的词频字典
posDict = {}
with open('D:\\pythonProject5\\1_Word_Segmentation\\C01-3seg1.txt', 'r', encoding='utf-8') as rmrbfile:
    for l in rmrbfile:
        # 取得每行的结果，以空格作为间隔
        for pstr in l.split(' '):
            # 去除每行的空格，换行符，得到每个词语和词性，列表。
            p = pstr.split('/')
            # 得到词语，不同词性的词语
            if len(p)>1:
                if posDict.get(p[0]):
                    posDict[p[0]].append(p[1])
                    posDict[p[0]] = list(set(posDict[p[0]]))
                else:
                    posDict[p[0]] = [p[1]]

#进行排序
posList = []
for key in posDict.keys():
    posList.append([key, posDict[key]])
# 打印所有列表
print(sorted(posList, key=lambda item:len(item[1]), reverse=True))
# 打印前十个
posTop10 = sorted(posList, key=lambda item:len(item[1]), reverse=True)[:10]
for pos in posTop10:
    print(pos)

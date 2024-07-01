import pandas as pd
import os

os.chdir(r'D:\pythonProject5\5_wordNet')
edge_str = pd.read_csv('edge.csv',encoding='gbk')
edge_str.shape

edge_str1 = edge_str[edge_str['Weight']>3]
edge_str1.shape

Source = edge_str1['Source'].tolist()
Target = edge_str1['Target'].tolist()
co = Source + Target
co =list(set(co))

node_str = pd.read_csv('node.csv',encoding='gbk')
#node_str

node_str=node_str[node_str['Label'].isin(co)]
node_str['id']=node_str['Label']
node_str = node_str[['id','Label','Weight']] # 调整列顺序
#node_str

node_str.to_csv(path_or_buf="node.txt", index=False) # 写入csv文件
edge_str1.to_csv(path_or_buf="edge.txt", index=False) # 写入csv文件

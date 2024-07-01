# 本程序仅使用词向量，人家训练好的词向量
from gensim.test.utils import datapath, get_tmpfile
from  gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

# 加载词向量
glove_file = datapath('D:\\pythonProject5\\2_Word2vec\\glove.1B.txt')
word2vec_glove_file = get_tmpfile('word2vec_glove_file.1B.txt')
glove2word2vec(glove_file, word2vec_glove_file)

model = KeyedVectors.load_word2vec_format(word2vec_glove_file)
simily_word = model.most_similar('human')
print(simily_word)
# result = model.most_similar(positive=['woman', 'king'], negative=['man'], topn=1)
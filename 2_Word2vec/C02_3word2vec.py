"""
第二步，训练模型
"""
import logging
import os.path
import sys
import multiprocessing
from gensim.corpora import WikiCorpus
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

# 主程序入口
if __name__ == '__main__':
    # 获取程序名称
    program = os.path.basename(sys.argv[0])
    # 初始化日志记录器
    logger = logging.getLogger(program)
    # 配置日志输出格式
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    # 设置日志级别为INFO
    logging.root.setLevel(level=logging.INFO)
    # 记录程序开始运行的日志
    logger.info("running %s" % ' '.join(sys.argv))

    # 检查命令行参数数量是否足够
    if len(sys.argv) < 4:
        # 打印帮助信息并退出
        print(globals()['__doc__'] % locals())
        sys.exit(1)

    # 解析输入和输出文件路径
    inp, outp1, outp2 = sys.argv[1:4]

    # 使用WikiCorpus和Word2Vec训练模型
    # 参数包括：输入文件路径，词向量大小，窗口大小，出现次数最少的词频，使用的worker线程数
    model = Word2Vec(LineSentence(inp), vector_size=200, window=5, min_count=5, workers=multiprocessing.cpu_count())

    # 保存Word2Vec模型
    model.save(outp1)

    # 以非二进制格式保存词向量
    model.wv.save_word2vec_format(outp2, binary=False)

#使用方法
# 在cmd中输入如下代码即可
# python word2vec.py corpus.txt word2vec.model word2vec.vector
# python C02_3word2vec.py bridge_text.target.txt bridge_word2vec.model bridge_word2vec.vector
# python C02_3word2vec.py 01theFirstOne.target.txt 01theFirstOne.model 01theFirstOne.vector

############################################################################################################


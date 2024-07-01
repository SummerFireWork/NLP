"""
随机抽样
"""

import random
print (random.randint(1, 235770))

"""
计算样本量
"""

n0 = (1.96*1.96*0.25)/(0.05*0.05)
n1 = n0*1.8/0.95
print (n1)
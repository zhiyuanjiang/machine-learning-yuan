## ID3算法实现
* 先确定当前数据集上哪个特征在划分数据分类上起决定性作用。
* 通过计算熵来确定最佳划分特征(熵最大的)。
* $H = \sum_{i=1}^{n} P(xi)log(P(xi))$
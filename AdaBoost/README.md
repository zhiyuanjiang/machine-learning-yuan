@(机器学习)

### AdaBoost算法

AdaBoost算法使用多个弱分类器构建一个强分类器。

算法流程：
* 给训练数据中的每一个样本赋予一个权重，这些权重构成了一个向量D。
* 首先在这些数据上训练一个弱分类器，并计算错误率$\varepsilon$，$\varepsilon$的定义如下：
$$
\varepsilon = \frac{未正确分类的样本数目}{所有样本数}
$$
* 为分类器分配一个权重值$\alpha$，$\alpha$定义如下：
$$
\alpha = \frac{1}{2}ln(\frac{1-e}{e})
$$
* 更新权重向量D：
如果样本被正确分类
$$
D_{i}^{t+1} = \frac{D_{i}^{t}e^{-\alpha}}{sum(D)}
$$
如果样本被错误分类
$$
D_{i}^{t+1} = \frac{D_{i}^{t}e^{\alpha}}{sum(D)}
$$
* 继续在数据集上训练一个弱分类器，重复上述步骤（弱分类器的个数是一个超参数）。

![Alt text](./icon/Image.png)


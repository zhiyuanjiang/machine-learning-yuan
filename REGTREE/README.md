
### 树回归

#### 回归树
将叶节点设置为常数值。
CART算法：使用二元切分来处理连续性变量。

使用dict来存储回归树，包含下面几个特征：
* 待切分的特征
* 待切分的特征值
* 右子树，当不需要切分时也可以是单个的值
* 左子树

**算法伪代码**：
* 找到最佳的切分特征。
* 如果该节点不能再分，将该节点存为叶子节点。
* 执行二元切分。
* 左子树递归执行。
* 右子树递归执行。


**树剪枝(防止过拟合)**：
* 预剪枝 (在构建模型的时候就进行剪枝)
* 后剪枝 (先在训练集上训练出一个足够大的模型，然后用测试集判断将这些叶子节点合并能否降低测试误差)

#### 模型树
将叶节点设置为分段线性函数。

误差计算：
对于给定的数据集，先用线性回归的模型对它进行拟合，然后计算真实的目标值与模型预测值间的差值，最后这些差值的平方和就是误差。


#### 模型性能评估
计算$R^{2}$
可以调用numpy中的**corrcoef(yHat, y, rowvar=0)**来计算

**树回归在预测复杂数据时比简单的线性模型更加有效**。
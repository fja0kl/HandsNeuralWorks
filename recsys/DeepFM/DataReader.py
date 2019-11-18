import pandas as pd
'''
FeatureDictionary: 学习feature2idx字典. 
    - 如果当前特征是被忽略特征,跳过;
    - 如果是连续特征,idx+1;
    - 如果是离散特征,当前特征的每个取值都当做一个新的特征,idx+len(unique_values)[相当于对离散特征onehot化,但是我们在向量表示时并不取全部的维度]

DataParser: 将数据转换格式. pandas dataframe转换成向量稀疏表示: Xi, Xv两个数组来表示对应的数据
    - Xi表示当前特征的在feature2idx的编号
    - Xv表示对应的取值情况. 如果当前特征是连续特征,则取值为原始值;如果是离散特征,对应取值为1.
'''
class FeatureDictionary:
    """
    获取原始数据集转换后特征字典[原始特征->转换后特征映射关系],以及特征数.
    """
    def __init__(self,trainfile=None,testfile=None,
                 dfTrain=None,dfTest=None,numeric_cols=[],
                 ignore_cols=[]):
        assert not ((trainfile is None) and (dfTrain is None)), "trainfile or dfTrain at least one is set"
        assert not ((trainfile is not None) and (dfTrain is not None)), "only one can be set"
        assert not ((testfile is None) and (dfTest is None)), "testfile or dfTest at least one is set"
        assert not ((testfile is not None) and (dfTest is not None)), "only one can be set"

        self.trainfile = trainfile # file 和 df只能设置一个,不同同时设置
        self.testfile = testfile
        self.dfTrain = dfTrain
        self.dfTest = dfTest
        self.numeric_cols = numeric_cols
        self.ignore_cols = ignore_cols
        self.gen_feat_dict()

    def gen_feat_dict(self):
        if self.dfTrain is None:# 设置file, 读取file获得dataframe对象
            dfTrain = pd.read_csv(self.trainfile)

        else:
            dfTrain = self.dfTrain

        if self.dfTest is None:
            dfTest = pd.read_csv(self.testfile)

        else:
            dfTest = self.dfTest

        df = pd.concat([dfTrain,dfTest], axis=0)# 训练集和测试集同时学习映射字典和特征数

        self.feat_dict = {}
        tc = 0
        for col in df.columns:
            if col in self.ignore_cols: # 忽略特征,略过
                continue
            if col in self.numeric_cols: # 数值特征,不变
                self.feat_dict[col] = tc
                tc += 1

            else: # 类别离散特征
                us = df[col].unique() # 获得当前原始特征的取值情况, one-hot后,当前特征的每个取值作为一个新的特征
                print('feature {}, value list: {}'.format(col, us))
                self.feat_dict[col] = dict(zip(us,range(tc,len(us)+tc))) # 原始特征onehot转换后,对应的特征编号范围
                tc += len(us)

        self.feat_dim = tc # 新特征维度


class DataParser:
    """
    数据集转化器: 将原始特征表示数据,转换/压缩后的特征表示情况.
    压缩表示格式为: 两个列表,一个表示转换后的特征编号id,一个表示取值情况.
    """
    def __init__(self,feat_dict):
        self.feat_dict = feat_dict

    def parse(self,infile=None,df=None,has_label=False):
        assert not ((infile is None) and (df is None)), "infile or df at least one is set"
        assert not ((infile is not None) and (df is not None)), "only one can be set"


        if infile is None:
            dfi = df.copy()
        else:
            dfi = pd.read_csv(infile)

        if has_label:
            y = dfi['target'].values.tolist()
            dfi.drop(['id','target'],axis=1,inplace=True)
        else:
            ids = dfi['id'].values.tolist()
            dfi.drop(['id'],axis=1,inplace=True)
        # dfi for feature index
        # dfv for feature value which can be either binary (1/0) or float (e.g., 10.24)
        dfv = dfi.copy()
        for col in dfi.columns:
            if col in self.feat_dict.ignore_cols:# 忽略的特征列,删去
                dfi.drop(col,axis=1,inplace=True)
                dfv.drop(col,axis=1,inplace=True)
                continue
            if col in self.feat_dict.numeric_cols:# 数值特征,不做变化,直接取id编号
                dfi[col] = self.feat_dict.feat_dict[col]
            else: # 离散特征 
                dfi[col] = dfi[col].map(self.feat_dict.feat_dict[col])# 根据原始特征的取值情况,在新特征转换字典中找到对应的编号[字典当前特征, 对应的值也是字典,查找]
                dfv[col] = 1.# 取值是1.

        xi = dfi.values.tolist()
        xv = dfv.values.tolist()
        print("xi type:", type(xi), len(xi), len(xi[0]))
        if has_label:
            return xi,xv,y
        else:
            return xi,xv,ids

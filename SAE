import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from scipy.linalg import solve_sylvester

path = '/Users/zhuxiaoxiansheng/Desktop/Animals_with_Attributes2/'

classname = pd.read_csv(path+'classes.txt',header=None,sep = '\t')
dic_class2name = {classname.index[i]:classname.loc[i][1] for i in range(classname.shape[0])}
dic_name2class = {classname.loc[i][1]:classname.index[i] for i in range(classname.shape[0])}
    
def make_test_attributetable():
    attribut_bmatrix = pd.read_csv(path+'predicate-matrix-continuous-01.txt',header=None,sep = ',')
    test_classes = pd.read_csv(path+'testclasses.txt',header=None)
    test_classes_flag = []
    for item in test_classes.iloc[:,0].values.tolist():
        test_classes_flag.append(dic_name2class[item])
    return attribut_bmatrix.iloc[test_classes_flag,:]
        
trainlabel = np.load(path+'AWA2_trainlabel.npy')
train_attributelabel = np.load(path+'AWA2_train_continuous_01_attributelabel.npy')

testlabel = np.load(path+'AWA2_testlabel.npy')
test_attributelabel = np.load(path+'AWA2_test_continuous_01_attributelabel.npy')

trainfeatures = np.load(path+'resnet101_trainfeatures.npy')
testfeatures = np.load(path+'resnet101_testfeatures.npy')

print(trainfeatures.shape,testfeatures.shape)

lam = 7

S = np.mat(train_attributelabel).T
X = np.mat(trainfeatures).T

A = S*S.T
B = lam*X*X.T
C = (1+lam)*S*X.T
W = solve_sylvester(A,B,C)

test_pre_attribute = (W*(np.mat(testfeatures).T)).T  
print(test_pre_attribute.shape)
test_attributetable = make_test_attributetable()

label_lis = []
for i in range(test_pre_attribute.shape[0]):
    pre_res = test_pre_attribute[i,:]
    loc = np.sum(np.square(test_attributetable.values - pre_res),axis=1).argmin()
    label_lis.append(test_attributetable.index[loc])

print(accuracy_score(list(testlabel),label_lis))


















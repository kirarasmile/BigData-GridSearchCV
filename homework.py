# -*- coding: utf-8 -*-
# 信用卡违约率分析
import pandas as pd
from sklearn.model_selection import learning_curve, train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier
from matplotlib import pyplot as plt
import seaborn as sns

# data load
data = pd.read_csv('./UCI_Credit_Card.csv')

# data discovery
# print(data.shape)
# print(data.describe())

# 查看下个月违约率的情况
next_month = data['default.payment.next.month'].value_counts()
# print(next_month)
df = pd.DataFrame({'default.payment.next.month': next_month.index, 'values': next_month.values})
plt.rcParams['font.sans-serif'] = ['SimHei']# 显示中文标签
plt.figure(figsize = (6, 6))
plt.title('违约客户率\n (违约：1， 守约：0)')
sns.set_color_codes("pastel")
sns.barplot(x = 'default.payment.next.month', y = "values", data = df)
locas , labels = plt.xticks()
plt.show()
# 特征选择，去掉ID字段、最后一个结果字段
data.drop(['ID'], inplace = True, axis = 1)
target = data['default.payment.next.month'].values
columns = data.columns.tolist()
columns.remove('default.payment.next.month')
features = data[columns].values
# 30%作为测试集，其余作为训练集
train_x, test_x, train_y, test_y = train_test_split(features, target, test_size = 0.30, stratify = target, random_state = 1)

# 构造分类器
ada = AdaBoostClassifier(random_state=1)


# 分类器参数
parameters = {'adaboostclassifier__n_estimators':[10,50,100]}
# parameters = {'n_estimators':[10,50,100]}

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('adaboostclassifier', ada)
])

# 进行GridSearchCV参数调优

clf = GridSearchCV(estimator = pipeline, param_grid = parameters, scoring = 'accuracy')
# clf = GridSearchCV(estimator = ada, param_grid = parameters, scoring = 'accuracy')
clf.fit(train_x, train_y)
print("最优参数:", clf.best_params_)
print("最优分数: %0.4lf" %clf.best_score_)
predict_y = clf.predict(test_x)
print("准确率%0.4lf" %accuracy_score(test_y, predict_y))




# 分类任务
# 创建分类器
# sklearn.metrics
def train_model(classifier, train_feature, train_label, test_feature, test_label):
    classifier.fit(train_feature, train_label)
    prediction = classifier.predict(test_feature)
    acc = metrics.accuracy_score(prediction, test_label)
    prec = metrics.precision_score(test_label, prediction, average='weighted')
    rec = metrics.recall_score(test_label, prediction, average='weighted')
    f1 = metrics.f1_score(test_label, prediction, average='weighted')
    return acc, prec, rec, f1

# 5.1 朴素贝叶斯多项式模型
# 5.1.1 计数特征向量
# Sklearn.naive_bayes
# accuracy, precision, recall, f1_socre = train_model(naive_bayes.MultinomialNB(), xtrain_count, xtest_count)
# print("NB, Count Vectors: ", accuracy)
accuracy, precision, recall, f1_score = train_model(naive_bayes.MultinomialNB(), xtrain_count, train_label, xtest_count, test_label)
print("MulNB, Count Vectors: ", accuracy, precision, recall, f1_score)

model = naive_bayes.MultinomialNB()
model.fit(xtrain_count, train_label)
# 使用测试集的特征向量进行预测
predictions = model.predict(xtest_count)
# 使用测试集的标签来计算评估指标
accuracy = metrics.accuracy_score(test_label, predictions)
precision = metrics.precision_score(test_label, predictions, average='weighted')
recall = metrics.recall_score(test_label, predictions, average='weighted')
f1_score = metrics.f1_score(test_label, predictions, average='weighted')
print("MulNB, Count Vectors: ", accuracy, precision, recall, f1_score)
# 保存模型
joblib.dump(model, 'MulNB_model.pkl')
# 保存向量化器
joblib.dump(count_vect, 'MulNB_vectorizer.pkl')


import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
import numpy as np

# 初始化模型
model = MultinomialNB()
# 使用learning_curve函数来获取训练集大小、训练分数和验证分数
train_sizes, train_scores, validation_scores = learning_curve(
    model, xtrain_count, train_label, cv=5, scoring='accuracy', n_jobs=-1, 
    train_sizes=np.linspace(0.01, 1.0, 50))
# 计算平均和标准差
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
validation_scores_mean = np.mean(validation_scores, axis=1)
validation_scores_std = np.std(validation_scores, axis=1)
# 绘制学习曲线
plt.figure()
plt.title('Learning Curve')
plt.xlabel('Training examples')
plt.ylabel('Score')
# 绘制训练分数曲线
plt.plot(train_sizes, train_scores_mean, label='Training score', color='r', marker='o')
plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, color='r', alpha=0.1)
# 绘制验证分数曲线
# plt.ylim([0.9, 1.0])
plt.plot(train_sizes, validation_scores_mean, label='Cross-validation score', color='g', marker='o')
plt.fill_between(train_sizes, validation_scores_mean - validation_scores_std, validation_scores_mean + validation_scores_std, color='g', alpha=0.1)
plt.legend(loc='best')
plt.show()

# 伯努利模型
model = naive_bayes.BernoulliNB()
model.fit(xtrain_count, train_label)
# 使用测试集的特征向量进行预测
predictions = model.predict(xtest_count)
# 使用测试集的标签来计算评估指标
accuracy = metrics.accuracy_score(test_label, predictions)
precision = metrics.precision_score(test_label, predictions, average='weighted')
recall = metrics.recall_score(test_label, predictions, average='weighted')
f1_score = metrics.f1_score(test_label, predictions, average='weighted')
print("BernliNB, Count Vectors: ", accuracy, precision, recall, f1_score)
# 保存模型
joblib.dump(model, 'BernliNB_model.pkl')
# 保存向量化器
joblib.dump(count_vect, 'BernliNB_vectorizer.pkl')

from sklearn.naive_bayes import ComplementNB

model = ComplementNB()
model.fit(xtrain_count, train_label)
# 使用测试集的特征向量进行预测
predictions = model.predict(xtest_count)
# 使用测试集的标签来计算评估指标
accuracy = metrics.accuracy_score(test_label, predictions)
precision = metrics.precision_score(test_label, predictions, average='weighted')
recall = metrics.recall_score(test_label, predictions, average='weighted')
f1_score = metrics.f1_score(test_label, predictions, average='weighted')
print("CompleNB, Count Vectors: ", accuracy, precision, recall, f1_score)
# 保存模型
joblib.dump(model, 'CompleNB_model.pkl')
# 保存向量化器
joblib.dump(count_vect, 'CompleNB_vectorizer.pkl')

# 支持向量机模型
model = svm.SVC()
model.fit(xtrain_count, train_label)
# 使用测试集的特征向量进行预测
predictions = model.predict(xtest_count)
# 使用测试集的标签来计算评估指标
accuracy = metrics.accuracy_score(test_label, predictions)
precision = metrics.precision_score(test_label, predictions, average='weighted')
recall = metrics.recall_score(test_label, predictions, average='weighted')
f1_score = metrics.f1_score(test_label, predictions, average='weighted')
print("SVM, Count Vectors: ", accuracy, precision, recall, f1_score)
# 保存模型
joblib.dump(model, 'SVM_model.pkl')
# 保存向量化器
joblib.dump(count_vect, 'SVM_vectorizer.pkl')

# 随机森林模型
model = ensemble.RandomForestClassifier()
model.fit(xtrain_count, train_label)
# 使用测试集的特征向量进行预测
predictions = model.predict(xtest_count)
# 使用测试集的标签来计算评估指标
accuracy = metrics.accuracy_score(test_label, predictions)
precision = metrics.precision_score(test_label, predictions, average='weighted')
recall = metrics.recall_score(test_label, predictions, average='weighted')
f1_score = metrics.f1_score(test_label, predictions, average='weighted')
print("RFCF, Count Vectors: ", accuracy, precision, recall, f1_score)
# 保存模型
joblib.dump(model, 'RFCF_model.pkl')
# 保存向量化器
joblib.dump(count_vect, 'RFCF_vectorizer.pkl')

# KNN模型
model = neighbors.KNeighborsClassifier()
model.fit(xtrain_count, train_label)
# 使用测试集的特征向量进行预测
predictions = model.predict(xtest_count)
# 使用测试集的标签来计算评估指标
accuracy = metrics.accuracy_score(test_label, predictions)
precision = metrics.precision_score(test_label, predictions, average='weighted')
recall = metrics.recall_score(test_label, predictions, average='weighted')
f1_score = metrics.f1_score(test_label, predictions, average='weighted')
print("KNN, Count Vectors: ", accuracy, precision, recall, f1_score)
# 保存模型
joblib.dump(model, 'KNN_model.pkl')
# 保存向量化器
joblib.dump(count_vect, 'KNN_vectorizer.pkl')


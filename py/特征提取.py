import json
with open('data/preprocessed_data/preprocessed_trec07p_emails.json', 'r', encoding='utf-8') as jsonfile:
    all_emails = json.load(jsonfile)

with open('preprocessed_trec07p_labels_emails.json', 'r', encoding='utf-8') as jsonfile:
    labels = json.load(jsonfile)

import pandas
# 特征提取
# 创建一个dataframe，列名为text和label
trainDF = pandas.DataFrame()
trainDF['text'] = all_emails
trainDF['label'] = labels

# 将数据集分为训练集和测试集，以便模型能在训练集上学习并在测试集上验证其性能
# sklearn.model_selection.train_test_split
train_data, test_data, train_label, test_label = train_test_split(trainDF['text'],trainDF['label'], random_state=0)

# label编码为目标变量,即从字符串转为一个数字
# sklearn.preprocessing
encoder = preprocessing.LabelEncoder()
train_label = encoder.fit_transform(train_label)
test_label = encoder.fit_transform(test_label)

trainDF['text'] = [' '.join(email) for email in all_emails]
train_data = [' '.join(doc) for doc in train_data]
test_data = [' '.join(doc) for doc in test_data]

# 计数特征向量
# sklearn.feature_extraction.text.CountVectorizer
count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
count_vect.fit(trainDF['text'])
xtrain_count = count_vect.transform(train_data)  # 训练集特征向量
xtest_count = count_vect.transform(test_data)  # 测试集特征向量

# TF-IDF特征向量
# sklearn.feature_extraction.text.TfidfVectorizer
# 词语级
tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
# 多词语级
tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2, 3), max_features=5000)
# 词性级
tfidf_vect_char = TfidfVectorizer(analyzer='char', ngram_range=(2, 3), max_features=5000)
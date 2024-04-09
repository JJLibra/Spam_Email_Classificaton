import os
import re
from collections import Counter
from email import parser, policy
from html import unescape

import nltk
import pandas
import urlextract
from nltk.corpus import stopwords
from sklearn import metrics, preprocessing, naive_bayes
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split

from email.parser import BytesParser

import joblib
# import sys
# print(sys.executable)

# 读取数据集
INDEX_PATH = os.path.join('trec07p', 'full', 'index')  # 先使用较小的数据集进行训练
DATA_PATH = os.path.join('trec07p', 'data')  # 数据文件夹路径
labels = []
filenames = []

# 将delay中的标签和对应的文件名保存
def create_dataset(index_path):
    with open(index_path) as f:
        while True:
            line = f.readline()
            if not line:
                break
            line = line.split(' ')
            labels.append(line[0])
            filenames.append(line[1].strip('\n').split('/')[-1])
            
create_dataset(INDEX_PATH)

# 下载delay中索引的文件内容
def load_email(filename, file_path):
    with open(os.path.join(file_path, filename), 'rb') as f:
        return parser.BytesParser(policy=policy.default).parse(f)

raw_emails = [load_email(name, DATA_PATH) for name in filenames]

print(raw_emails[3].get_content().strip())  # 打印邮件文本内容，注意这里输出内容不一定是正确的，只有当邮件为文本类型，才能输出；不能正确输出说明邮件是多部份的

# 数据预处理
# 构造函数获取邮件的结构类型及其计数

# 确定每封电子邮件的结构类型
def get_email_structure(email):
    if isinstance(email, str): # 字符串直接返回
        return email
    payload = email.get_payload() # 提取email的主体部分
    if isinstance(payload, list): # 如果是列表，说明该邮件为多部份
        return 'multipart({})'.format(', '.join([get_email_structure(sub_email) for sub_email in payload]))
        # 如果有两个子部分，一个是 text/plain，另一个是 text/html，那么最终的结果将是 'multipart(text/plain, text/html)'
    else: # 否则一般是text/plain或text/html
        return email.get_content_type()

# 统计一组电子邮件中各种结构类型的出现次数
def structures_counter(emails):
    structures = Counter() # 字典类型
    for email in emails:
        structure = get_email_structure(email)
        structures[structure] += 1
    return structures

print(structures_counter(raw_emails).most_common())  # 显示邮件包含的类型
"""
这对于分析电子邮件数据集的结构分布非常有用，可以帮助我们理解数据集中最常见的电子邮件类型，从而为进一步的数据处理和特征工程提供信息。
例如，如果多数邮件都是纯文本类型，那么我们可能会专注于文本内容的分析；如果有大量的多部分邮件，我们可能需要考虑如何处理嵌入的图片或附件。
"""

# 将原始的电子邮件内容转换为更适合文本分析和机器学习模型训练的格式

# 将HTML内容转换为纯文本，同时替换为相应的关键词
def html_to_plain_text(html):
    text = re.sub('<head.*?>.*?</head>', '', html, flags=re.M | re.S | re.I)
    text = re.sub(r'<[aA]\s.*?>', 'HYPERLINK', text, flags=re.M | re.S | re.I)
    text = re.sub(r'<img\s.*?>', 'IMAGE', text, flags=re.M | re.S | re.I)
    text = re.sub('<.*?>', '', text, flags=re.M | re.S)
    text = re.sub(r'(\s*\n)+', '\n', text, flags=re.M | re.S)
    return unescape(text)

# 用于从电子邮件中提取文本内容
def email_to_text(email):
    html = None
    # walk()打印出一封具有多部分结构之信息的每个部分的MIME类型
    for part in email.walk():
        ctype = part.get_content_type()
        if ctype not in ('text/plain', 'text/html'):
            continue
        try:
            content = part.get_content()
        except LookupError:
            content = str(part.get_payload())
        if ctype == 'text/plain':
            return content
        else:
            html = content
    if html:
        return html_to_plain_text(html)

# 分词
stopwords_list = stopwords.words('english')  # 英文停用词列表
token = nltk.stem.SnowballStemmer('english')  # 提取词干，词干提取器对象，用于将单词还原为基本形式或词干。例如，将“running”还原为“run”。

# 将所有小写英文字母添加到停用词列表
for single in range(97, 123):
    stopwords_list.append(chr(single))
    
extractor = urlextract.URLExtract() # 创建了一个URL提取器对象，用于从文本中找出URL

# 将电子邮件文本转换为一个清洗和标准化的单词列表
def word_split(email):
    text = email_to_text(email) or ' '
    text = text.lower()
    text = re.sub(r'\W+', ' ', text, flags=re.M) # 使用正则表达式替换文本中的所有非字母数字字符为单个空格
    urls = list(set(extractor.find_urls(text))) # 一个去重的URL列表
    urls.sort(key=lambda item: len(item), reverse=True) # 将找到的URL按长度降序排序
    for url in urls:
        text = text.replace(url, "URL") # 将文本中的所有URL替换为特征词“URL”
    text = re.sub(r'\d+(?:\.\d*[eE]\d+)?', 'NUMBER', text) # 使用正则表达式将文本中的所有数字替换为字符串“NUMBER”
    content = list(nltk.word_tokenize(text)) # 使用NLTK的 word_tokenize 函数将文本分割成单词列表
    all_words = []
    for word in content:
        if word not in stopwords_list:
            word = token.stem(word)
            all_words.append(word)
    return all_words

all_emails = [word_split(data) for data in raw_emails]
print(all_emails[1])  # 查看分词结果

# 保存处理结果，方便后面直接使用
import json
# 将预处理后的邮件数据保存到JSON文件
with open('preprocessed_trec07p_emails.json', 'w', encoding='utf-8') as jsonfile:
    json.dump(all_emails, jsonfile, ensure_ascii=False, indent=4)


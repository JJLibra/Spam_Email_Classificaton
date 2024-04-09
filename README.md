# 🤖 Spam_Email_Classificaton

## Traditional machine learning

## Bayesian theory

P(A│B)=(P(B│A)·P(A))/P(B) ，P(B)≠0

After introducing the naive assumption：

P(C│F_1,F_2,…,F_n )=(∏P(F_i│C)·P(C))/(P(F_1 )·P(F_2 )·…·P(F_n ) )∝P(C)∏P(F_i│C)，i=1,2,…,n

## Polynomial Naive Bayes Model

### Accuracy&Precision&Recall&F1 Score

| Metric    | Value  |
|-----------|--------|
| Accuracy  | 0.9475 |
| Precision | 0.9533 |
| Recall    | 0.9475 |
| F1 Score  | 0.9482 |

The performance metrics of the Polynomial Model are summarized as follows: The model achieves an Accuracy of 94.75%, indicating a high overall rate of correct predictions. The Precision, which measures the model's ability to correctly identify positive instances, is 95.33%, demonstrating the model's strong predictive accuracy for positive classifications. The Recall or Sensitivity, indicating the model's capability to find all the relevant cases within a dataset, is also 94.75%, showing the model's effectiveness in identifying all positive samples. Lastly, the F1 Score, which is the harmonic mean of Precision and Recall, stands at 94.82%, suggesting a balanced performance between Precision and Recall. Overall, these metrics indicate a robust performance of the Polynomial Model across various evaluation criteria.

### Confusion matrix and learning curve

<div>
  <img src="./img/MulNB1.png" alt="混淆矩阵" style="width: 45%; height: 350px; float: left;">
  <img src="./img/MulNB4.png" alt="学习曲线" style="width: 45%; height: 350px; float: right;">
</div>
<div style="clear: both;"></div>

The polynomial naive Bayes model performs well in robustness and does not suffer from overfitting.
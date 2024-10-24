import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

bank_processed = pd.read_csv('bank/bank_processed.csv', sep=';')

# 特征相关性分析
def feature_correlation(df):
    # 只保留数值列
    numeric_df = df.select_dtypes(include=[np.number])
    correlation_matrix = numeric_df.corr()
    return correlation_matrix

# 可视化
def plot_correlation_heatmap(correlation_matrix):
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Feature Correlation Heatmap')
    plt.show()

# 展示不同特征的分布情况
def plot_feature_distribution(df, feature):
    plt.figure(figsize=(10, 6))
    sns.countplot(x=feature, hue='y', data=df)
    plt.title(f'Distribution of {feature} by Response')
    plt.show()

# 模型分析
def logistic_regression_analysis(df):
    df = pd.get_dummies(df, drop_first=True)
    # 选择特征和目标变量
    X = df.drop('y', axis=1)
    y = df['y']
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # 拆分数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 训练逻辑回归模型
    model = LogisticRegression(max_iter=2000)
    model.fit(X_train, y_train)

    # 预测
    y_pred = model.predict(X_test)

    # 打印分类报告和混淆矩阵
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # 返回模型和特征重要性
    return model, model.coef_


def analysis(df):
    # 计算特征相关性
    correlation_matrix = feature_correlation(df)
    print("Feature Correlation Matrix:\n", correlation_matrix)

    # 可视化特征相关性
    plot_correlation_heatmap(correlation_matrix)

    # 展示特征分布情况
    features_to_plot = ['age', 'job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'poutcome']
    for feature in features_to_plot:
        plot_feature_distribution(df, feature)

    # 模型分析
    model, feature_importance = logistic_regression_analysis(df)
    print("Feature Importance:\n", feature_importance)

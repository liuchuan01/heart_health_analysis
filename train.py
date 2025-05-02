import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
import joblib

def load_data():
    """加载数据并进行预处理"""
    df = pd.read_csv('data/heart.csv')
    
    # 处理分类特征
    categorical_features = ['ChestPain', 'Thal']
    df = pd.get_dummies(df, columns=categorical_features)
    
    return df

def prepare_features(df):
    """准备特征和目标变量"""
    X = df.drop('Target', axis=1)
    y = df['Target']
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 特征标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 保存scaler供后续使用
    joblib.dump(scaler, 'models/scaler.pkl')
    
    return X_train_scaled, X_test_scaled, y_train, y_test

def get_models():
    """获取所有可用的模型"""
    models = {
        'random_forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        'svm': SVC(probability=True, random_state=42),
        'logistic_regression': LogisticRegression(random_state=42),
        'naive_bayes': GaussianNB(),
        'decision_tree': DecisionTreeClassifier(random_state=42),
        'neural_network': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42),
        'knn': KNeighborsClassifier(n_neighbors=5)
    }
    return models

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """训练并评估所有模型"""
    models = get_models()
    results = {}
    
    for name, model in models.items():
        print(f"\n训练模型: {name}")
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        print(f"模型准确率: {score:.4f}")
        
        # 保存模型
        joblib.dump(model, f'models/{name}_model.pkl')
        results[name] = score
    
    # 保存模型评估结果
    joblib.dump(results, 'models/model_scores.pkl')
    return results

def main():
    # 加载数据
    df = load_data()
    
    # 准备特征
    X_train, X_test, y_train, y_test = prepare_features(df)
    
    # 训练和评估所有模型
    results = train_and_evaluate_models(X_train, X_test, y_train, y_test)
    
    # 打印最终结果
    print("\n所有模型评估结果:")
    for name, score in results.items():
        print(f"{name}: {score:.4f}")

if __name__ == "__main__":
    main() 
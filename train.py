import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
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

def train_model(X_train, y_train):
    """训练模型"""
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """评估模型"""
    score = model.score(X_test, y_test)
    print(f"Model accuracy: {score:.4f}")
    return score

def main():
    # 加载数据
    df = load_data()
    
    # 准备特征
    X_train, X_test, y_train, y_test = prepare_features(df)
    
    # 训练模型
    model = train_model(X_train, y_train)
    
    # 评估模型
    score = evaluate_model(model, X_test, y_test)
    
    # 保存模型
    joblib.dump(model, 'models/model.pkl')

if __name__ == "__main__":
    main() 
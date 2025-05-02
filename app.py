import streamlit as st
import pandas as pd
import numpy as np
import joblib

def load_models():
    """加载所有预训练的模型和scaler"""
    models = {}
    model_names = ['random_forest', 'svm', 'logistic_regression', 
                  'naive_bayes', 'decision_tree', 'neural_network', 'knn']
    
    for name in model_names:
        try:
            models[name] = joblib.load(f'models/{name}_model.pkl')
        except:
            continue
    
    scaler = joblib.load('models/scaler.pkl')
    scores = joblib.load('models/model_scores.pkl')
    
    return models, scaler, scores

def predict(features, model, scaler):
    """使用选定的模型进行预测"""
    # 转换为DataFrame
    df = pd.DataFrame([features])
    
    # 处理分类特征
    chest_pain_cols = ['ChestPain_asymptomatic', 'ChestPain_nonanginal', 
                      'ChestPain_nontypical', 'ChestPain_typical']
    thal_cols = ['Thal_fixed', 'Thal_normal', 'Thal_reversable']
    
    # 初始化所有分类特征列为0
    for col in chest_pain_cols + thal_cols:
        df[col] = 0
    
    # 设置对应的分类特征为1
    df[f'ChestPain_{features["ChestPain"].lower()}'] = 1
    df[f'Thal_{features["Thal"].lower()}'] = 1
    
    # 删除原始分类列
    df = df.drop(['ChestPain', 'Thal'], axis=1)
    
    # 标准化特征
    df_scaled = scaler.transform(df)
    
    # 预测概率
    proba = model.predict_proba(df_scaled)[0]
    prediction = model.predict(df_scaled)[0]
    
    return prediction, proba[1]

def main():
    st.title('心脏病预测系统')
    st.write('请输入患者的健康指标，系统将预测心脏病风险。')
    
    # 加载模型
    models, scaler, scores = load_models()
    
    if not models:
        st.error('未找到训练好的模型，请先运行train.py训练模型！')
        return
    
    # 显示模型选择和准确率
    st.sidebar.header('模型选择')
    model_name = st.sidebar.selectbox(
        '选择预测模型',
        list(models.keys()),
        format_func=lambda x: {
            'random_forest': '随机森林',
            'svm': '支持向量机',
            'logistic_regression': '逻辑回归',
            'naive_bayes': '朴素贝叶斯',
            'decision_tree': '决策树',
            'neural_network': '神经网络',
            'knn': 'K近邻'
        }[x]
    )
    
    st.sidebar.write(f'模型准确率: {scores[model_name]:.2%}')
    
    # 用户输入
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input('年龄', min_value=1, max_value=100, value=45)
        sex = st.selectbox('性别', ['男', '女'])
        chest_pain = st.selectbox(
            '胸痛类型',
            ['typical', 'nontypical', 'nonanginal', 'asymptomatic'],
            format_func=lambda x: {
                'typical': '典型心绞痛',
                'nontypical': '非典型心绞痛',
                'nonanginal': '非心绞痛',
                'asymptomatic': '无症状'
            }[x]
        )
        rest_bp = st.number_input('静息血压', min_value=0, max_value=300, value=120)
        chol = st.number_input('胆固醇水平', min_value=0, max_value=600, value=200)
        
    with col2:
        fbs = st.selectbox('空腹血糖 > 120 mg/dl', ['是', '否'])
        rest_ecg = st.selectbox('心电图结果', [0, 1, 2], 
                              format_func=lambda x: ['正常', 'ST-T波异常', '左心室肥大'][x])
        max_hr = st.number_input('最大心率', min_value=0, max_value=300, value=150)
        exang = st.selectbox('运动诱发心绞痛', ['是', '否'])
        oldpeak = st.number_input('ST压低', min_value=0.0, max_value=10.0, value=0.0)
        slope = st.selectbox('ST段斜率', [0, 1, 2], 
                           format_func=lambda x: ['上升', '平坦', '下降'][x])
        ca = st.number_input('主要血管数量', min_value=0, max_value=4, value=0)
        thal = st.selectbox(
            'Thal',
            ['normal', 'fixed', 'reversable'],
            format_func=lambda x: {
                'normal': '正常',
                'fixed': '固定缺陷',
                'reversable': '可逆缺陷'
            }[x]
        )
    
    # 准备特征
    features = {
        'Age': age,
        'Sex': 1 if sex == '男' else 0,
        'ChestPain': chest_pain,
        'RestBP': rest_bp,
        'Chol': chol,
        'Fbs': 1 if fbs == '是' else 0,
        'RestECG': rest_ecg,
        'MaxHR': max_hr,
        'ExAng': 1 if exang == '是' else 0,
        'Oldpeak': oldpeak,
        'Slope': slope,
        'Ca': ca,
        'Thal': thal
    }
    
    if st.button('预测'):
        prediction, probability = predict(features, models[model_name], scaler)
        
        st.write('---')
        if prediction == 1:
            st.error(f'预测结果：可能患有心脏病（置信度：{probability:.2%}）')
        else:
            st.success(f'预测结果：心脏健康（置信度：{1-probability:.2%}）')
        
        # 显示风险因素分析
        if probability > 0.5:
            st.write('### 主要风险因素：')
            risk_factors = []
            if age > 60:
                risk_factors.append('年龄偏高')
            if rest_bp > 140:
                risk_factors.append('血压偏高')
            if chol > 200:
                risk_factors.append('胆固醇水平偏高')
            if max_hr > 180:
                risk_factors.append('最大心率偏高')
            if oldpeak > 2:
                risk_factors.append('ST压低程度显著')
            if ca >= 2:
                risk_factors.append('主要血管钙化数量较多')
            
            for factor in risk_factors:
                st.write(f'- {factor}')

if __name__ == '__main__':
    main() 
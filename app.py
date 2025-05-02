import streamlit as st
import pandas as pd
import numpy as np
import joblib

def load_models():
    """加载预训练的模型和scaler"""
    model = joblib.load('models/model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    return model, scaler

def predict(features, model, scaler):
    """使用模型进行预测"""
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
    
    # 预测
    prediction = model.predict(df_scaled)
    probability = model.predict_proba(df_scaled)
    
    return prediction[0], probability[0]

def main():
    st.title('心脏病预测系统')
    st.write('请输入以下信息进行心脏病风险预测：')
    
    # 创建输入表单
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input('年龄', min_value=20, max_value=100, value=50)
            sex = st.selectbox('性别', ['男', '女'])
            chest_pain = st.selectbox('胸痛类型', 
                ['typical', 'asymptomatic', 'nonanginal', 'nontypical'])
            rest_bp = st.number_input('静息血压', min_value=80, max_value=200, value=120)
            chol = st.number_input('胆固醇', min_value=100, max_value=600, value=200)
            fbs = st.selectbox('空腹血糖 > 120 mg/dl', ['是', '否'])
            
        with col2:
            rest_ecg = st.selectbox('心电图结果', [0, 1, 2])
            max_hr = st.number_input('最大心率', min_value=60, max_value=220, value=150)
            exang = st.selectbox('运动诱发心绞痛', ['是', '否'])
            oldpeak = st.number_input('ST压低', min_value=0.0, max_value=10.0, value=0.0)
            slope = st.selectbox('ST段斜率', [1, 2, 3])
            ca = st.selectbox('荧光检查着色血管数', [0, 1, 2, 3])
            thal = st.selectbox('地中海贫血', ['normal', 'fixed', 'reversable'])
            
        submitted = st.form_submit_button("预测")
    
    if submitted:
        # 处理输入数据
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
        
        # 加载模型并预测
        model, scaler = load_models()
        prediction, probability = predict(features, model, scaler)
        
        # 显示结果
        st.subheader('预测结果')
        if prediction == 1:
            st.error('⚠️ 可能患有心脏病')
            st.write(f'患病概率: {probability[1]:.2%}')
        else:
            st.success('✅ 心脏健康')
            st.write(f'健康概率: {probability[0]:.2%}')
        
        # 显示风险因素分析
        st.subheader('风险因素分析')
        risk_factors = []
        if age > 60:
            risk_factors.append('年龄偏高')
        if rest_bp > 140:
            risk_factors.append('血压偏高')
        if chol > 200:
            risk_factors.append('胆固醇偏高')
        if fbs == '是':
            risk_factors.append('空腹血糖偏高')
        
        if risk_factors:
            st.write('需要注意的风险因素：')
            for factor in risk_factors:
                st.write(f'- {factor}')
        else:
            st.write('未发现明显风险因素')

if __name__ == '__main__':
    main() 
# 心脏病预测系统

这是一个基于机器学习的心脏病预测系统，可以根据用户输入的各项健康指标预测心脏病风险。

## 功能特点

- 使用随机森林算法进行预测
- 提供友好的Web界面
- 实时预测结果
- 风险因素分析
- 预测概率显示

## 安装说明

1. 克隆项目到本地：
```bash
git clone [项目地址]
cd heart-disease-prediction
```

2. 创建并激活虚拟环境（推荐）：
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows
```

3. 安装依赖：
```bash
pip install -r requirements.txt
```

## 使用说明

1. 准备数据：
   - 将数据文件 `heart.csv` 放在 `data` 目录下

2. 训练模型：
```bash
python train.py
```

3. 启动Web应用：
```bash
streamlit run app.py
```

4. 在浏览器中访问应用（默认地址：http://localhost:8501）

## 项目结构

```
heart-disease-prediction/
├── data/
│   └── heart.csv
├── models/
│   ├── model.pkl
│   └── scaler.pkl
├── app.py
├── train.py
├── requirements.txt
└── README.md
```

## 输入特征说明

- Age: 年龄
- Sex: 性别（1=男性；0=女性）
- ChestPain: 胸痛类型
- RestBP: 静息血压
- Chol: 胆固醇水平
- Fbs: 空腹血糖
- RestECG: 心电图结果
- MaxHR: 最大心率
- ExAng: 运动诱发心绞痛
- Oldpeak: ST压低
- Slope: ST段斜率
- Ca: 荧光检查着色血管数
- Thal: 地中海贫血类型

## 注意事项

- 本系统仅供参考，不能替代专业医生的诊断
- 如有异常情况，请及时就医
- 定期进行体检，预防胜于治疗 
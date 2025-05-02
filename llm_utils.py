import os
from openai import OpenAI
from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, ANALYSIS_PROMPT

class LLMAnalyzer:
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv('DEEPSEEK_API_KEY', DEEPSEEK_API_KEY),
            base_url=os.getenv('DEEPSEEK_BASE_URL', DEEPSEEK_BASE_URL)
        )
    
    def format_patient_info(self, features):
        """格式化患者信息"""
        info = []
        translations = {
            'Age': '年龄',
            'Sex': '性别',
            'ChestPain': '胸痛类型',
            'RestBP': '静息血压',
            'Chol': '胆固醇水平',
            'Fbs': '空腹血糖',
            'RestECG': '心电图结果',
            'MaxHR': '最大心率',
            'ExAng': '运动诱发心绞痛',
            'Oldpeak': 'ST压低',
            'Slope': 'ST段斜率',
            'Ca': '主要血管数量',
            'Thal': 'Thal检查结果'
        }
        
        for key, value in features.items():
            if key == 'Sex':
                value = '男' if value == 1 else '女'
            elif key == 'Fbs':
                value = '是' if value == 1 else '否'
            elif key == 'ExAng':
                value = '是' if value == 1 else '否'
            elif key == 'RestECG':
                value = ['正常', 'ST-T波异常', '左心室肥大'][value]
            elif key == 'Slope':
                value = ['上升', '平坦', '下降'][value]
            
            info.append(f"{translations.get(key, key)}: {value}")
        
        return "\n".join(info)
    
    def get_health_analysis(self, features, prediction_result, model_name, confidence):
        """获取LLM健康分析"""
        patient_info = self.format_patient_info(features)
        
        prompt = ANALYSIS_PROMPT.format(
            patient_info=patient_info,
            prediction_result=prediction_result,
            model_name=model_name,
            confidence=confidence
        )
        
        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "你是一位经验丰富的心脏病专家，请根据患者信息给出专业的建议。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"无法获取AI分析结果：{str(e)}\n请检查API配置是否正确。"
    
    def get_health_analysis_stream(self, features, prediction_result, model_name, confidence):
        """获取LLM健康分析（流式输出）"""
        patient_info = self.format_patient_info(features)
        
        prompt = ANALYSIS_PROMPT.format(
            patient_info=patient_info,
            prediction_result=prediction_result,
            model_name=model_name,
            confidence=confidence
        )
        
        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "你是一位经验丰富的心脏病专家，请根据患者信息给出专业的建议。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1000,
                stream=True  # 启用流式输出
            )
            return response
        except Exception as e:
            raise Exception(f"无法获取AI分析结果：{str(e)}\n请检查API配置是否正确。") 
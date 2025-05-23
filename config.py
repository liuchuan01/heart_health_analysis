# LLM API配置
DEEPSEEK_API_KEY = "sk-0******d3f26df"  # 在环境变量中设置
DEEPSEEK_BASE_URL = "https://api.deepseek.com"  # 在环境变量中设置

# LLM提示词模板
ANALYSIS_PROMPT = """
请根据以下患者信息和预测结果，给出专业的健康建议总结：

患者信息：
{patient_info}

预测结果：
{prediction_result}

预测模型：{model_name}
预测置信度：{confidence:.2%}

请从以下几个方面进行分析和建议：
1. 整体健康状况评估
2. 主要风险因素分析
3. 具体的改善建议
4. 是否建议进一步检查
""" 

# src/models/evaluate.py

import joblib
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

def evaluate_model(model_path, features, labels):
    """
    加载模型并进行评估
    :param model_path: 模型文件路径
    :param features: 特征数据，numpy数组格式
    :param labels: 标签数据，numpy数组格式
    """
    # 加载模型
    model = joblib.load(model_path)

    # 进行预测
    predictions = model.predict(features)

    # 计算评估指标
    mse = mean_squared_error(labels, predictions)
    r2 = r2_score(labels, predictions)

    print(f"均方误差 (MSE): {mse}")
    print(f"R²得分: {r2}")

if __name__ == "__main__":
    # 示例数据，您需要根据实际数据替换成真实的特征和标签
    features = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])  # 示例特征
    labels = np.array([1, 2])  # 示例标签
    evaluate_model('writing_assessment_model.pkl', features, labels)

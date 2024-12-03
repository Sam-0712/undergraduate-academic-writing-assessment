import os
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from features.vocabulary import calculate_vocabulary_richness
from features.syntax import analyze_syntax_complexity
from features.coherence import evaluate_coherence
from features.theme_relevance import evaluate_theme_relevance  # 假设您要添加一个新的特征模块
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def prepare_features_and_labels(preprocessed_data):
    features = []
    labels = []

    for data in preprocessed_data.values():
        # 提取现有特征
        vocab_richness = calculate_vocabulary_richness(data['tokens'])  # 词汇丰富度
        syntax_complexity = analyze_syntax_complexity(data['sentences'])[0]  # 句法复杂性
        coherence_score = evaluate_coherence(data['tokens'], data['coherence_words'])  # 连贯性

        # 新增特征：主题相关性，需要用到的参数有标题、摘要、关键词和正文
        theme_relevance = ......

        # 合并所有特征
        feature_vector = [vocab_richness, syntax_complexity, coherence_score, theme_relevance]

        # 假设您有固定的评分或标签
        label = data['score']  # 从数据集中提取的目标值

        features.append(feature_vector)
        labels.append(label)

    return np.array(features), np.array(labels)

def train_model(features, labels):
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # 创建和训练模型
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 在测试集上评估模型
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print(f"均方误差 (MSE): {mse}")
    print(f"R²得分: {r2}")

    # 保存模型
    joblib.dump(model, 'writing_assessment_model.pkl')
    print("模型已保存为 'writing_assessment_model.pkl'")

if __name__ == "__main__":
    # 调用数据预处理函数，得到处理后的数据
    from utils.preprocessing import preprocess_data
    base_directory = os.path.dirname(os.path.abspath(__file__))
    data_directory = os.path.join(base_directory, '../data/raw/')  # 更新为原始数据目录
    stop_words_path = os.path.join(base_directory, '../stopwords.txt')
    model_choice = 'jieba'  # 选择的分词模型

    preprocessed_essays = preprocess_data(data_directory, stop_words_path, model_choice)

    # 准备特征和标签
    features, labels = prepare_features_and_labels(preprocessed_essays)

    # 训练模型
    train_model(features, labels)

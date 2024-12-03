import sys
import os
import tqdm

# 将 src 目录添加到系统路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../src'))
sys.path.append(project_root)

from utils.preprocessing import preprocess_data  # 绝对导入
from features.vocabulary import calculate_vocabulary_richness
from features.syntax import analyze_syntax_complexity
from features.coherence import evaluate_coherence
from features.theme_relevance import evaluate_theme_relevance
# data数据在根目录\data\raw下，stopwords.txt文件在根目录\src下
data_directory = os.path.join(project_root, '..', 'data', 'raw')  # 原始数据目录
stop_words_path = os.path.join(project_root, 'stopwords.txt')  # 停用词文件路径

# 分词模型
model_choice = 'jieba'  # 用户可以指定分词模型，例如 'jieba'、'hanlp'、'snownlp'

def load_data(ids):
    # 获取处理后的数据
    data = preprocess_data(data_directory, stop_words_path, model_choice)  # 从 preprocessing.py 中获取处理后的数据

    # 根据用户提供的 ID 过滤数据
    filtered_data = {id: data.get(id) for id in ids if id in data}
    return filtered_data

def extract_features(content):
    # 解包内容并计算特征
    tokens = content['tokens']
    sentences = content['sentences']
    title = content['title']
    author = content['author']
    curriculum = content['curriculum']
    date = content['date']
    abstract = content['abstract']
    keywords = content['keywords']
    coherence_words = content['coherence_words']
    word_count = content['word_count']
    
    # 计算特征
    vocab_richness = calculate_vocabulary_richness(tokens)[3]
    syntax_complexity = analyze_syntax_complexity(sentences)[0]
    coherence_score = evaluate_coherence(tokens, coherence_words)
    relevance_score = (evaluate_theme_relevance(title, keywords, sentences)[0] + 
                       10 * evaluate_theme_relevance(abstract, keywords, sentences)[1]) / 2

    return {
        'id': content.get('id'),  # 返回ID
        'title': title,
        'author': author,
        'curriculum': curriculum,
        'date': date,
        'word_count': word_count,
        'vocab_richness': vocab_richness,
        'syntax_complexity': syntax_complexity,
        'coherence_score': coherence_score,
        'relevance_score': relevance_score
    }

def main():
    try:
        # 询问用户输入所需的 id
        user_input = input("请输入需要的数据ID（用空格隔开）: ")
        ids = user_input.split()

        # 加载数据
        data = load_data(ids)

        # 如果输入为空
        if not ids:
            # 给出提示
            print("请输入需要的数据ID。")
            return
        else:
            for id in ids:
                # 打印数据信息
                if id in data:
                    print(f"ID: {id}，标题: {data[id]['title']}，作者: {data[id]['author']}，课程: {data[id]['curriculum']}，日期: {data[id]['date']}")
                else:
                    print(f"ID: {id}，没有找到对应的数据。")
    
    except Exception as e:
        print(f"发生错误: {e}")

if __name__ == "__main__":
    main()
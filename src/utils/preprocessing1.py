# 标准库
import os
import subprocess

# 第三方库
import jieba
import pandas as pd
import numpy as np

# 用户可配置参数
data_directory = '../data/raw/'  # 原始数据目录
stop_words_path = './stopwords.txt'  # 停用词文件路径
model_choice = 'jieba'  # 用户可以指定分词模型，例如 'jieba'、'hanlp'、'snownlp'

# 加载原始数据
def load_data(data_dir):
    """加载原始数据，返回文件内容的字典"""
    essays = {}
    
    # 获取项目根目录
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, data_dir)  # 获取绝对路径

    # 检查目录是否存在
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"数据目录 '{data_dir}' 不存在，请检查路径是否正确。")
    
    try:
        for filename in os.listdir(data_dir):
            if filename.endswith('.txt'):
                with open(os.path.join(data_dir, filename), 'r', encoding='utf-8') as file:
                    essays[filename[:-4]] = file.read()  # 去除文件扩展名
    except Exception as e:
        raise RuntimeError(f"读取文件时发生错误: {e}")

    return essays

# 清洗文本
def clean_text(text):
    # 这里可以添加更多清洗规则
    text = text.replace('\n', '')  # 去除换行符
    return text

# 多模型分词
def tokenize(text, model='jieba'):
    if model == 'jieba':
        return list(jieba.cut(text))
    
    elif model == 'hanlp': # 使用 Python 3.8 的环境来调用 pyhanlp
        python38_executable = r"C:\Python38\python.exe"
        command = [python38_executable, '-c', f"import pyhanlp; print(pyhanlp.HanLP.segment('{text}').toString())"]
        output = subprocess.check_output(command, encoding='utf-8')  # 指定编码为 utf-8
        return output.strip().split()
    
    elif model == 'snownlp':
        from snownlp import SnowNLP
        s = SnowNLP(text)
        return list(s.words)

    raise ValueError("Unsupported model. Please choose 'jieba', 'hanlp', or 'snownlp'.")

# 去停用词
def remove_stop_words(tokens, stop_words_file):
    with open(stop_words_file, 'r', encoding='utf-8') as file:
        stop_words = set(file.read().splitlines())
    return [token for token in tokens if token not in stop_words]

# 预处理数据
def preprocess_data(data_dir, stop_words_file, model='jieba'):
    essays = load_data(data_dir)
    preprocessed_data = {}

    for title, text in essays.items():
        cleaned_text = clean_text(text)
        tokens = tokenize(cleaned_text, model)
        tokens = remove_stop_words(tokens, stop_words_file)
        preprocessed_data[title] = tokens  # 保存分词后的结果

    return preprocessed_data

if __name__ == "__main__":
    preprocessed_essays = preprocess_data(data_directory, stop_words_path, model_choice)

    for title, tokens in preprocessed_essays.items():
        print(f"Title: {title}, Tokens: {tokens}")

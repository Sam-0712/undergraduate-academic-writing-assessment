import os
import re
import sys
import io
import warnings
import subprocess
import xml.etree.ElementTree as ET
import logging
from gensim import corpora
from gensim.models import LdaModel
import pyLDAvis.gensim_models
import pyLDAvis
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)

# from features.vocabulary import calculate_vocabulary_richness

# 输入编码
sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')

# 目录，获取绝对路径
base_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_directory = './data/raw'
stop_words_path = './dict/stopwords.txt'
coherence_keywords_path = './dict/coherence_keywords.txt'

# 分词模型
model_choice = 'jieba'  # 用户可以指定分词模型，例如 'jieba'、'hanlp'、'snownlp'

# 禁止jieba在终端打印building和loading信息
warnings.filterwarnings('ignore', category=UserWarning, module='jieba')
warnings.filterwarnings("ignore", category=DeprecationWarning)

def load_data(data_dir):
    essays = {}
    data_dir = os.path.abspath(data_dir)
    print("数据目录:", data_dir)
    print("开始读取数据...")

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"数据目录 '{data_dir}' 不存在，请检查路径是否正确。")

    try:
        for filename in os.listdir(data_dir):
            if filename.endswith('.xml'):
                try:
                    tree = ET.parse(os.path.join(data_dir, filename))
                    root = tree.getroot()
                    for paper in root.findall('.//Paper'):
                        id = paper.find('ID').text
                        title = paper.find('Metadata/Title').text
                        author = paper.find('Metadata/Author').text
                        curriculum = paper.find('Metadata/Curriculum').text
                        year = paper.find('Metadata/Year').text
                        month = paper.find('Metadata/Month').text
                        day = paper.find('Metadata/Day').text
                        keywords = paper.find('Keywords').text
                        abstract = paper.find('Abstract').text
                        body = paper.find('Body').text
                        essays[id] = {
                            'title': title,
                            'author': author,
                            'curriculum': curriculum,
                            'date': f"{year}-{month}-{day}",
                            'keywords': keywords,
                            'abstract': abstract,
                            'body': body
                        }
                except ET.ParseError as e:
                    print(f"解析 {filename} 时发生错误: {e}")
                except Exception as e:
                    print(f"处理文件 '{filename}' 时发生错误: {e}")
    except Exception as e:
        raise RuntimeError(f"读取文件时发生错误: {e}")

    return essays

def split_sentences(text):
    text = re.sub(r'\s{2,}', ' ', text.replace('\n', ''))
    sentences = re.findall(r'[^。！？\s]+[。！？]?', text)
    return sentences, text

# 在分词之前，提前先处理一些数据
def preprocess_in_advance(text, tokens, word_count):
    ##### 功能一：统计关联词 #####
    # 关联词
    with open(coherence_keywords_path, 'r', encoding='utf-8') as f:
        coherence_dict = set(f.read().splitlines())
        coherence_words = [word for word in tokens if word in coherence_dict]
        coherence_parameters = tuple([len(coherence_words), round(len(coherence_words) / word_count,4)])

    ##### 功能二：转述性标记 #####
    # 转述性标记
    token_researches = ['研究', '分析', '采用', '定义', '出版', '提供', '界定', '阐述', '考察', '描述', '发现', '表明', '证明', '揭示', '证实', '得出', '显示', '发表', '推动', '开创'] # 研究性标记
    token_statements = ['认为', '提出', '指出', '强调', '所说', '探讨', '论述', '介绍', '主张'] # 话语性标记
    token_cognition = ['看来', '考虑', '承认', '接受', '当作', '相信', '关注', '注意到', '看作', '思考'] # 认知性标记

    # 统计各标记的出现次数
    token_researches_count = sum([1 for token in tokens if token in token_researches])
    token_statements_count = sum([1 for token in tokens if token in token_statements])
    token_cognition_count = sum([1 for token in tokens if token in token_cognition])
    token_total_count = token_statements_count + token_researches_count + token_cognition_count
    
    # 频率的计算，保留到小数点后四位
    token_researches_ratio = round(token_researches_count / word_count, 4)
    token_statements_ratio = round(token_statements_count / word_count, 4)
    token_cognition_ratio = round(token_cognition_count / word_count, 4)
    token_total_ratio = round(token_total_count / word_count, 4)

    # 按顺序保存频次和频率
    frequencies_counts = tuple([token_researches_count, token_statements_count, token_cognition_count, token_total_count,token_researches_ratio, token_statements_ratio, token_cognition_ratio,token_total_ratio])

    ##### 功能三：统计词汇密度（也就是实词的比例） #####
    # 使用jieba进行词性标注，统计实词的比例
    from jieba import posseg
    wordcutted = posseg.cut(text)
    real_word_count = len([word.word for word in wordcutted if word.flag in ['a', 'v', 'n', 't', 'nr', 'ns', 'nt', 'nz']])
    realword_ratio = round(real_word_count / len(tokens), 4)

    return coherence_words, coherence_parameters, frequencies_counts, realword_ratio

def tokenize(text, model):
    if model == 'jieba':
        import jieba
        jieba.setLogLevel(logging.ERROR)
        return list(jieba.cut(text))
    
    elif model == 'hanlp':
        python38_executable = r"C:\Python38\python.exe"
        command = [
            python38_executable,
            '-c',
            f"import pyhanlp; result = pyhanlp.HanLP.segment('{text}'); print(','.join([str(item.word) for item in result]))"
        ]
        try:
            output = subprocess.check_output(command, encoding='utf-8')
        except UnicodeDecodeError:
            output = subprocess.check_output(command, encoding='gbk')
        return [word.strip() for word in output.split(',')]
    
    elif model == 'snownlp':
        from snownlp import SnowNLP
        s = SnowNLP(text)
        return list(s.words)

    raise ValueError("Unsupported model. Please choose 'jieba', 'hanlp', or 'snownlp'.")

def preprocess_data(data_dir, stop_words_file, model, enable_lda_analysis):
    essays = load_data(data_dir)
    preprocessed_data = {}
    
    try:
        with open(stop_words_file, 'r', encoding='utf-8') as file:
            stop_words = set(file.read().splitlines())
    except Exception as e:
        raise RuntimeError(f"读取停用词文件时发生错误: {e}")

    for id, info in essays.items():
        sentences = split_sentences(info['body'])[0]
        cleaned_text = split_sentences(info['body'])[1]
        tokens = tokenize(cleaned_text, model)
        word_count = len(cleaned_text.replace(' ', '').replace('\n', ''))

        coherence_words, coherence_parameters, frequencies_counts, realword_ratio = preprocess_in_advance(info['body'], tokens, word_count)

        # 另一种实现方法：使用jieba进行词性标注，找出里面词性为c的词就是conjunctions_words
        # from jieba import posseg
        # words = posseg.cut(info['body'])
        # conjunctions_words = [word.word for word in words if word.flag == 'c']
        # conjunctions_count = len(conjunctions_words)
        # conjunctions_ratio = round(conjunctions_count / word_count, 4)

        keywords = re.findall(r'\w+', info['keywords'])
        abstract = re.sub(r'\s{2,}', ' ', info['abstract'].replace('\n', ''))

        # 去除停用词和空格，得到清洗后的token序列
        tokens = [token.strip() for token in tokens if token.strip() not in stop_words]

        # 添加基本的预处理结果
        preprocessed_data[id] = {
            'title': info['title'],
            'author': info['author'],
            'curriculum': info['curriculum'],
            'date': info['date'],
            'keywords': keywords,
            'abstract': abstract,
            'tokens': tokens,
            'sentences': sentences,
            'coherence_words': coherence_words,
            'coherence_parameters': coherence_parameters,
            'word_count': word_count,
            'frequencies_counts': frequencies_counts,
            'realword_ratio': realword_ratio
        }

        # 如果开启LDA分析，则执行LDA主题分析
        if enable_lda_analysis:
            # 构建字典和语料库
            dictionary = corpora.Dictionary([tokens])
            corpus = [dictionary.doc2bow(tokens)]

            # 训练LDA模型
            num_topics = 5
            lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=15)

            # 输出主题
            topic_keywords = []
            for idx, topic in lda_model.print_topics(-1):
                topic_keywords.append(f"Topic {idx}: {topic}")

            # 保存LDA可视化结果
            vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)
            save_path = os.path.join(data_directory, f'lda_visualization_{id}.html')
            pyLDAvis.save_html(vis, save_path)

            # 将主题关键词结果添加到preprocessed_data中
            preprocessed_data[id]['lda_topics'] = topic_keywords  # 添加LDA主题关键词字段

    return preprocessed_data

# 要打印结果作为测试
if __name__ == '__main__':
    enable_lda_analysis = False
    preprocessed_data = preprocess_data(data_directory, stop_words_path, model_choice, enable_lda_analysis)
    # print(preprocessed_data)

    # 打印某篇特定文档的数据，具体打印token、句子、主题关键词、coherence_words、word_count
    userinput = input("请输入要打印的文档ID：")
    if userinput in preprocessed_data:
        print(f"标题：{preprocessed_data[userinput]['title']}")
        print(f"句子：{preprocessed_data[userinput]['sentences']}")
        print(f"关联词：{preprocessed_data[userinput]['coherence_parameters']}")
        print(f"实词比例：{preprocessed_data[userinput]['realword_ratio']}")
        print(f"转述标记使用情况：{preprocessed_data[userinput]['frequencies_counts']}")
                
        tokens = preprocessed_data[userinput]['tokens']
        # 计算这篇文档的STTR，按照1000词一段的方式遍历，舍去最后不足1000词的部分
        sttr = 0
        num_segments = 0    
        if not tokens:
            print("文档为空，无法计算STTR。")
            
        for i in range(0, len(tokens) - len(tokens) % 1000, 1000):
            sub_tokens = tokens[i:i + 1000]  # 获取每 1000 个词
            sub_unique_tokens = set(sub_tokens)  # 获取唯一的词汇
            sub_ttr = len(sub_unique_tokens) / len(sub_tokens)  # 计算该段的 TTR
            sttr += sub_ttr
            num_segments += 1  # 有效的分段数量

        # 如果有效分段数大于 0，则计算平均值
        if num_segments > 0:
            sttr /= num_segments
            print(f"STTR：{sttr}")
        else:
            print("文档过短，无法计算STTR。")
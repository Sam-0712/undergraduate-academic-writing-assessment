import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import jieba  # 中文分词
import logging  # 禁止jieba在终端打印building和loading信息

def extract_keywords(sentence):
    # 禁止jieba在终端打印building和loading信息
    logging.getLogger('jieba').setLevel(logging.ERROR)

    # 使用结巴分词提取关键信息
    words = jieba.cut(sentence)
    return ' '.join(words)  # 用空格连接单词

def evaluate_theme_relevance(title, keywords, sentences):
    # 提取关键信息
    key_sentences = [extract_keywords(sentence) for sentence in sentences]
    key_title = extract_keywords(title)

    # 计算标题与每个句子的相似度
    documents = [key_title] + key_sentences
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)
    similarity_matrix = cosine_similarity(tfidf_matrix)
    
    # 提取标题-句子相似度
    title_sentence_similarities = similarity_matrix[0, 1:]  # 第一行是标题与所有句子的相似度

    # 计算关键词与每个句子的相似度
    keyword_similarities = []
    for keyword in keywords:
        keyword_vector = extract_keywords(keyword)
        keyword_documents = [keyword_vector] + key_sentences
        keyword_tfidf_matrix = vectorizer.fit_transform(keyword_documents)
        keyword_similarity_matrix = cosine_similarity(keyword_tfidf_matrix)

        # 提取关键词与句子的相似度
        keyword_sentence_similarities = keyword_similarity_matrix[0, 1:]  # 获取关键词与所有句子的相似度
        keyword_similarities.append(keyword_sentence_similarities)
    
    # 计算关键词与句子的平均相似度
    avg_keyword_similarities = np.mean(np.mean(keyword_similarities, axis=0))

    # 计算标题-句子相似度的最大10%数据的平均值
    num_top = int(len(title_sentence_similarities) * 0.1) or 1  # 计算10%的数量，至少为1
    top_title_scores = np.sort(title_sentence_similarities)[-num_top:]  # 找到最大10%的相似度值
    average_title_similarity = np.mean(top_title_scores)

    return average_title_similarity, avg_keyword_similarities

if __name__ == "__main__":
    # 示例数据
    title = "学术写作能力的提高"
    keywords = ["学术", "写作", "大学生", "能力"]
    sentences = [
        "学术写作是一个复杂的过程。",
        "随着时间的推移，大学生的写作能力不断提高。",
        "本文将探讨影响写作能力的因素。"
    ]
    
    # 计算主题相关性
    scores = evaluate_theme_relevance(title, keywords, sentences)
    print(f"标题与句子的相似度: {scores[0]:.4f}, "
          f"关键词与句子的平均相关性: {scores[1]:.4f}")

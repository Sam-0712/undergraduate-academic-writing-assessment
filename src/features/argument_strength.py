# src/features/argument_strength.py

from collections import Counter
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
import jieba
import jieba.posseg

# 计算论文的论点强度
def evaluate_argument_strength(tokens, sentences):
    # 统计词频
    token_counter = Counter(tokens)

    # 用set去重
    unique_tokens = set(tokens)

    # 这里的top_n是不重复词数量的1%
    top_n = int(len(token_counter) * 0.01)
    most_common = token_counter.most_common(top_n)  # 获取高频词

    # 提取高频关键词
    keywords = [word for word, _ in most_common]

    # 计算高频关键词的出现次数
    keyword_frequency = sum(token_counter[keyword] for keyword in keywords)

    # 计算关键词覆盖率
    words_coverage_ratio = keyword_frequency / len(tokens) if len(tokens) > 0 else 0

    # 计算所有含有关键词的句子占总句子数的比例
    sentence_coverage_ratio = sum(1 for sentence in sentences if any(keyword in sentence for keyword in keywords)) / len(sentences)
    # 计算所有含有关键词的句子占总字数的比例
    passage_coverage_ratio = sum(len(sentence) for sentence in sentences if any(keyword in sentence for keyword in keywords)) / sum(len(sentence) for sentence in sentences)

    return keywords, words_coverage_ratio, sentence_coverage_ratio, passage_coverage_ratio

from snownlp import SnowNLP

def analyze_argument_style(sentences):
    argument_counts = [0, 0, 0, 0]  # [强支持, 弱支持, 强反对, 弱反对]
    sentiment_scores = []  # 用于保存每个句子的情感分数

    for sentence in sentences:
        s = SnowNLP(sentence)
        sentiment_score = s.sentiments  # 获取情感分数，范围为0到1
        sentiment_scores.append(sentiment_score)  # 保存情感分数

        # 根据情感分数更新计数器
        if sentiment_score >= 0.8:  # 强支持
            argument_counts[0] += 1
        elif sentiment_score >= 0.5:  # 弱支持
            argument_counts[1] += 1
        elif sentiment_score <= 0.2:  # 强反对
            argument_counts[2] += 1
        elif sentiment_score <= 0.5:  # 弱反对
            argument_counts[3] += 1

    # 将情感分数列表转换为元组
    sentiment_scores_tuple = tuple(sentiment_scores)
    
    return argument_counts, sentiment_scores_tuple  # 返回计数和情感分数元组

# 以下为测试代码
if __name__ == '__main__':
    sentences = ['一、《洛神赋》之情感简析', '文学作品的意蕴是文学作品结构的最深层次，它包含审美情韵、历史内容和哲学意涵三个维度，而这三个维度并不是平行排列的。', '文学作品所包含的情感与主旨不可混为一谈，但我们会看到历来论家对《洛神赋》的分析一贯遵循的思路是“主旨——情感”，即先确定曹植写作《洛神赋》的目的，以此作为预设来分析曹植所想要表达的情感。', '这条思路基于曹植自己所写的直接型“创作动机”（“感宋玉对楚王神女之事”）和未写在文本之中的曹植经历，初看起来是显性的；但这一想法本质上是两汉经学家“文以载道”（《通书·文辞》）想法的延续：儒家认为，文学作品必须有教育意义，要么批评现实，要么抒发内心情 感，歌颂也勉强可以被归入“寄托”之中。', '在我看来，这一观念是有待考量的。', '文学创作的整体过程中，第一位的便是产生创作动因，它包括创作动机和创作冲动。', '创作动机大致相当于通俗意义上的“创作目的”，这种内驱力是个体性因素和社会性因素的合力。', '但创作冲动在文学批评中却常常被忽略，这可能是因为创作冲动大多转瞬即逝、难以言明，限于知识水平古代评论家又很难对这种心理过程进行系统性描写。', '但恰恰是这种创作冲动推动着作家进行文学创作，而这种冲动又是与客观固有的明确创作动机相独立的。', '“无寄托说”亦是一种对《洛神赋》主旨的观点，但此处并非必须肯定或者否定“无寄托说”，而是先不去讨论曹植是否确有寄托，单论其在语词上所体现的情绪，并把文章的主旨视为对情绪归纳得出的结果。', '同时我们也应该看到《洛神赋》中另外一位主角洛神自身也是有情感波动的，这就与曹植自身的情感相映成趣。', '《洛神赋》全篇大致可分为六个段落，分别对应曹植和洛神不同的情感：', '曹植', '洛神', '1.还济洛川', '疲惫心烦', '（尚未出现）', '2.初见洛神', '惊为天人', '（恬然自得）', '3.心有所思', '振荡不怡', '4.犹豫狐疑', '犹豫迷茫', '5.互通心意', '矜持守礼', '心生爱慕', '悲伤痛苦', '6.永诀不见', '怅惘盘桓', '心系君王', '以上的分析仅仅来自于对文本抒情（包括直接抒情和间接抒情）的归纳。', '整个故事的主题是“人神之恋”，其中“楔子”部分洛神还没有出现（此处洛神的引入是通过“御者”完成的，但是其主要影响叙事结构，对整体的情感表达影响不大），描写洛神绝色的几段也并没有明确写洛神之情感。', '自“指潜渊而为期”开始，曹植方与洛神产生对话，文章中洛神之情感是曹植“托微波而通辞”之后才产生的。', '这样，至少就文本而言情感的产生就有了先后之分。', '作为情感产生主动的一方，曹植的情感是非常丰富的，“车殆马烦”之后的种种情感可以大致归为喜悦和悲伤两个侧面。']
    argument_counts, sentiment_scores = analyze_argument_style(sentences)
    print("论证风格计数:", argument_counts)  # 输出计数结果
    print("情感分数元组:", sentiment_scores)  # 输出情感分数元组


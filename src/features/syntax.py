# src/features/syntax.py

def analyze_syntax_complexity(sentences):
    total_length = 0
    sentence_count = 0  # 句子计数
    clause_count = 0  # 从句计数

    for sentence in sentences:
        # 计算句子长度
        total_length += len(sentence)
        sentence_count += 1
        
        # 统计从句
        clause_count += sentence.count('，') + sentence.count('；')

    if sentence_count == 0:
        avg_sentence_length = 0
        clause_count = 0

    avg_sentence_length = total_length / sentence_count  # 计算平均句子长度

    # 句子丰富度
    avg_clause_per_sentence = clause_count / sentence_count

    # 返回平均句子长度
    return avg_sentence_length, avg_clause_per_sentence

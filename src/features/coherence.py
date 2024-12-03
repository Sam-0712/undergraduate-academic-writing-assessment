# src/features/coherence.py

def evaluate_coherence(tokens, coherence_words):
    # 这两个已经在preprocessed_data中定义了，这里直接调用即可
    coherence_score = len(coherence_words) / len(tokens) if tokens else 0  # 计算连贯性得分
    return coherence_score
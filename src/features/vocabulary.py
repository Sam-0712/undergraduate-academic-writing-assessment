# src/features/vocabulary.py

def calculate_vocabulary_richness(tokens):
    if not tokens:
        return 0

    # 计算词汇总数
    total_tokens = len(tokens)
    # 计算词汇种类数
    unique_tokens = len(set(tokens))
    ttr = unique_tokens / total_tokens

    # 计算标准化的词汇丰富度
    sttr = 0
    num_segments = 0  # 记录有效的分段数

    # 按 1000 词一段的方式遍历 tokens，但是考虑到论文长度，改为500
    for i in range(0, len(tokens) - len(tokens) % 500, 500):
        sub_tokens = tokens[i:i + 500]  # 获取每 500 个词
        sub_unique_tokens = set(sub_tokens)  # 获取唯一的词汇
        sub_ttr = len(sub_unique_tokens) / len(sub_tokens)  # 计算该段的 TTR
        sttr += sub_ttr
        num_segments += 1  # 有效的分段数量

    # 如果有效分段数大于 0，则计算平均值
    if num_segments > 0:
        sttr /= num_segments
    else:
        sttr = "null"  # 无效分段，返回 N/A

    # 返回标准化的词汇丰富度
    return total_tokens, unique_tokens, ttr, sttr
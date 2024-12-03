from tqdm import tqdm
from utils.preprocessing import preprocess_in_advance, preprocess_data # 数据传输到预处理转换为字段
from features.vocabulary import calculate_vocabulary_richness
from features.syntax import analyze_syntax_complexity
from features.coherence import evaluate_coherence
from features.theme_relevance import evaluate_theme_relevance
from features.argument_strength import evaluate_argument_strength # analyze_keywords
import warnings, os, sys

# 忽略警告
warnings.filterwarnings("ignore", category=RuntimeWarning)

# 用户配置的路径和设置
data_directory = './data/raw/'
stop_words_path = './dict/stopwords.txt'
model_choice = 'jieba'

# 打印表格
def print_table_row(id, title, author, curriculum, date, word_count, vocab_richness, realword_ratio, syntax_complexity, clause_density, coherence_score, frequencies_score, relevance_score, keyratio_score):
    global max_title_length, max_curriculum_length
    max_title_length = 36
    max_curriculum_length = 14
    
    # 表格打印辅助函数
    def calculate_length(text):
        length = 0
        fullwidth_punctuations = ['！', '‘', '’', '（', '）', '《', '》', '，', '。', '、', '；', '：', '？']
        for char in text:
            if ord(char) > 255 or char in fullwidth_punctuations:  # Unicode或全角字符
                length += 2
            else:
                length += 1 
            if char == '—' or char == '“' or char == '”':
                length += -1
        return length
    
    # 设置截断
    if calculate_length(title) > max_title_length:
        title = title[:max_title_length - 22] + "..."
        if calculate_length(title) > max_title_length:
            title = title[:max_title_length]

    if calculate_length(curriculum) > max_curriculum_length:
        curriculum = curriculum[:max_curriculum_length - 9] + "..."
        if calculate_length(curriculum) > max_curriculum_length:
            curriculum = curriculum[:max_curriculum_length]

    title_length = calculate_length(title)
    spaces_after_title = max(max_title_length - title_length, 0)

    curriculum_length = calculate_length(curriculum)
    spaces_after_curriculum = max(max_curriculum_length - curriculum_length, 0)

    vocab_richness_str = f"{vocab_richness:.4f}" if isinstance(vocab_richness, (float, int)) else vocab_richness
    realword_ratio_str = f"{realword_ratio:.4f}" if isinstance(realword_ratio, (float, int)) else realword_ratio
    # 如果syntax_complexity小于10，则保留3位小数后，在整个字符串最后加上"？"
    syntax_complexity_str = f"{syntax_complexity:.3f}?" if syntax_complexity < 10 else f"{syntax_complexity:.3f}"
    return ("│" + str(id).center(2) + "│" +
          title + ' ' * spaces_after_title + "│" +  # author.center(3) + "│" + # 隐私保护暂时屏蔽字段
          date.center(10) + "│" +
          curriculum + ' ' * spaces_after_curriculum + "│" +
          str(word_count).center(6) + "│" +
          vocab_richness_str.center(8) + "│" +
          realword_ratio_str.center(8) + "│" +
          syntax_complexity_str.center(8) + "│" +
          f"{clause_density:.4f}".center(8) + "│" +
          f"{coherence_score:.4f}".center(8) + "│" +
          f"{frequencies_score:.4f}".center(8) + "│" +
          f"{relevance_score:.4f}".center(8) + "│" +
          f"{keyratio_score:.4f}".center(8) + "│" )

################ 输出格式说明 #######
# 【WC】字数：文章的总字数。
# 【STTR】词汇丰富度：文章的总词汇数除以总字数。
# 【RW-R】实词比：文章实词数除以总词数（去除停用词前）。
# 【SL】平均句长：文章的平均句长。
# 【Co-R】连贯性：文章整体的连贯程度。
# 【RM-R】转述标记比例：文章中转述标记的比例，衡量观点引用等。
# 【T-Re】主题相关度：文章主题相关度，即文章有多切题。
# 【Str.】论证强度：文章强论证的整体幅度。
################ 输出格式说明 #######

# 主函数
def main():
    enable_lda_analysis = False  # 是否启用LDA主题分析，默认关闭
    preprocessed_essays = preprocess_data(data_directory, stop_words_path, model_choice, enable_lda_analysis)

    results = []
    global processed_papers_count
    processed_papers_count = 0
    print("论文数据已全部载入，正在计算特征...")

    for id, data in tqdm(preprocessed_essays.items(), desc="Processing papers", unit="papers", bar_format='{l_bar}{bar:40}{r_bar}', ncols=100):
        fields = ['title', 'author', 'curriculum', 'date', 'abstract', 'keywords', 'tokens', 'sentences', 'coherence_words', 'coherence_parameters', 'word_count', 'frequencies_counts', 'realword_ratio']
        title, author, curriculum, date, abstract, keywords, tokens, sentences, coherence_words, coherence_parameters, word_count, frequencies_counts, realword_ratio = (data[field] for field in fields)

        vocab_richness = calculate_vocabulary_richness(tokens)[3]
        # vocab_density = calculate_vocabulary_density(tokens)
        syntax_complexity, clause_density = analyze_syntax_complexity(sentences)
        relevance_score = (evaluate_theme_relevance(title, keywords, sentences)[0] + 10 * evaluate_theme_relevance(abstract, keywords, sentences)[1])/2

        keywords, words_coverage_ratio, sentence_coverage_ratio, passage_coverage_ratio = evaluate_argument_strength(tokens, sentences)
        keyratio_score = evaluate_argument_strength(tokens, sentences)[2] # 也就是words_coverage_ratio

        # 生成行数据并存储在列表中
        result_row = print_table_row(id, title, author, curriculum, date, word_count, vocab_richness, realword_ratio, syntax_complexity, clause_density, coherence_parameters[1], frequencies_counts[7], relevance_score, keyratio_score)
        results.append(result_row)

        # 处理的论文计数
        processed_papers_count += 1

    # 打印表头
    print("特征计算完毕，详细结果如下：")
    print("╭" + "─" * 2 + "┬" + "─" * max_title_length + "┬" + "─" * 10 + "┬" + "─" * max_curriculum_length + "┬" + "─" * 6 + "┬" + "─" * 8 + "┬" + "─" * 8 + "┬" + "─" * 8 + "┬" + "─" * 8 + "┬" + "─" * 8 + "┬" + "─" * 8 + "┬" + "─" * 8 + "┬" + "─" * 8 + "╮")
    print("│" + "ID".center(2) + "│" + "Title".center(max_title_length) + "│" + "Date".center(10) + "│" + "Curriculum".center(max_curriculum_length) + "│" + "WC".center(6) + "│" + "STTR".center(8) + "│" + "RW-R".center(8) + "│" + "SL".center(8) + "│" + "Clause".center(8) + "│" + "Co-R".center(8) + "│" + "RM-R".center(8) + "│" + "T-Re".center(8) + "│" + "Str.".center(8) + "│")
    print("╞" + "═" * 2 + "╪" + "═" * max_title_length + "╪" + "═" * 10 + "╪" + "═" * max_curriculum_length + "╪" + "═" * 6 + "╪" + "═" * 8 + "╪" + "═" * 8 + "╪" + "═" * 8 + "╪" + "═" * 8 + "╪" + "═" * 8 + "╪" + "═" * 8 + "╪" + "═" * 8 + "╪" + "═" * 8 + "╡")

    # 打印所有结果行
    for row in results:
        print(row)

    # 打印表格底部
    print("╰" + "─" * 2 + "┴" + "─" * max_title_length + "┴" + "─" * 10 + "┴" + "─" * max_curriculum_length + "┴" + "─" * 6 + "┴" + "─" * 8 + "┴" + "─" * 8 + "┴" + "─" * 8 + "┴" + "─" * 8 + "┴" + "─" * 8 + "┴" + "─" * 8 + "┴" + "─" * 8 + "┴" + "─" * 8 + "╯")

# 程序入口
if __name__ == "__main__":
    # 开始计时
    import time
    start_time = time.time()

    # 运行主函数
    main()

    # 结束计时
    end_time = time.time()
    print("程序运行完毕！处理论文", processed_papers_count, "篇，用时", "{:.2f}".format(end_time - start_time), "秒，平均速度", "{:.2f}".format(processed_papers_count / (end_time - start_time)), "篇/秒。")

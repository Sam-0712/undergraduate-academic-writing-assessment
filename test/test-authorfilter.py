import os, sys
from tqdm import tqdm
import numpy as np

# 返回上一目录作为根目录
# 将 src 目录添加到系统路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../src'))
sys.path.append(project_root)

# 导入所需的包
from utils.preprocessing import preprocess_data
from features.vocabulary import calculate_vocabulary_richness
from features.syntax import analyze_syntax_complexity
from features.coherence import evaluate_coherence
from features.theme_relevance import evaluate_theme_relevance
from features.argument_strength import evaluate_argument_strength
import warnings

# 忽略警告
warnings.filterwarnings("ignore", category=RuntimeWarning)

# 用户配置的路径和设置
data_directory = './data/raw/'
stop_words_path = './dict/stopwords.txt'
model_choice = 'jieba'

# 打印表格行
def print_table_row(id, title, author, curriculum, date, word_count, vocab_richness, realword_ratio, syntax_complexity, clause_density, coherence_score, frequencies_score, relevance_score, keyratio_score):
    global max_title_length, max_curriculum_length
    max_title_length = 36
    max_curriculum_length = 14
    
    # 辅助函数
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
    syntax_complexity_str = f"{syntax_complexity:.3f}?" if syntax_complexity < 10 else f"{syntax_complexity:.3f}"
    return ("│" + str(id).center(5) + "│" +
          title + ' ' * spaces_after_title + "│" +
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
          f"{keyratio_score:.4f}".center(8) + "│")

# 主函数
def main():
    enable_lda_analysis = False
    preprocessed_essays = preprocess_data(data_directory, stop_words_path, model_choice, enable_lda_analysis)

    results = []
    global processed_papers_count, target_author
    processed_papers_count = 0

    print("论文数据已全部载入，正在计算特征...")
    target_author = input("请输入要分析的论文作者：")

    # 存储指标
    all_metrics = {
        "word_count": [],
        "vocab_richness": [],
        "sentence_length": [],
        "clause_density": [],
        "coherence_score": [],
        "reporting-markers_ratio": [],
        "relevance_score": [],
        "argument_strength_score": []
    }

    for id, data in tqdm(preprocessed_essays.items(), desc="Processing papers", unit="papers", bar_format='{l_bar}{bar:40}{r_bar}', ncols=100):
        fields = ['title', 'author', 'curriculum', 'date', 'abstract', 'keywords', 'tokens', 'sentences', 'coherence_words', 'coherence_parameters', 'word_count', 'frequencies_counts', 'realword_ratio']
        title, author, curriculum, date, abstract, keywords, tokens, sentences, coherence_words, coherence_parameters, word_count, frequencies_counts, realword_ratio = (data[field] for field in fields)

        # 仅处理目标作者的论文
        if author == target_author:
            vocab_richness = calculate_vocabulary_richness(tokens)[3]
            syntax_complexity, clause_density = analyze_syntax_complexity(sentences)
            relevance_score = (evaluate_theme_relevance(title, keywords, sentences)[0] + 10 * evaluate_theme_relevance(abstract, keywords, sentences)[1]) / 2

            keywords, words_coverage_ratio, sentence_coverage_ratio, passage_coverage_ratio = evaluate_argument_strength(tokens, sentences)
            keyratio_score = evaluate_argument_strength(tokens, sentences)[2]  # 也就是words_coverage_ratio

            # 存储指标
            all_metrics["word_count"].append(word_count)
            all_metrics["vocab_richness"].append(vocab_richness)
            all_metrics["sentence_length"].append(syntax_complexity)
            all_metrics["clause_density"].append(clause_density)
            all_metrics["coherence_score"].append(coherence_parameters[1])
            all_metrics["reporting-markers_ratio"].append(frequencies_counts[7])
            all_metrics["relevance_score"].append(relevance_score)
            all_metrics["argument_strength_score"].append(keyratio_score)

            # 生成行数据并存储在列表中
            result_row = print_table_row(id, title, author, curriculum, date, word_count, vocab_richness, realword_ratio, syntax_complexity, clause_density, coherence_parameters[1], frequencies_counts[7], relevance_score, keyratio_score)
            results.append(result_row)

            # 处理的论文计数
            processed_papers_count += 1

    # 输出结果
    if results:
        print("特征计算完毕，详细结果如下：")
        print("╭" + "─" * 5 + "┬" + "─" * max_title_length + "┬" + "─" * 10 + "┬" + "─" * max_curriculum_length + "┬" + "─" * 6 + "┬" + "─" * 8 + "┬" + "─" * 8 + "┬" + "─" * 8 + "┬" + "─" * 8 + "┬" + "─" * 8 + "┬" + "─" * 8 + "┬" + "─" * 8 + "┬" + "─" * 8 + "╮")
        print("│" + "PID".center(5) + "│" + "Title".center(max_title_length) + "│" + "Date".center(10) + "│" + "Curriculum".center(max_curriculum_length) + "│" + "WC".center(6) + "│" + "STTR".center(8) + "│" + "RW-R".center(8) + "│" + "SL".center(8) + "│" + "Clause".center(8) + "│" + "Co-R".center(8) + "│" + "RM-R".center(8) + "│" + "T-Re".center(8) + "│" + "Str.".center(8) + "│")
        print("╞" + "═" * 5 + "╪" + "═" * max_title_length + "╪" + "═" * 10 + "╪" + "═" * max_curriculum_length + "╪" + "═" * 6 + "╪" + "═" * 8 + "╪" + "═" * 8 + "╪" + "═" * 8 + "╪" + "═" * 8 + "╪" + "═" * 8 + "╪" + "═" * 8 + "╪" + "═" * 8 + "╪" + "═" * 8 + "╡")

        # 打印所有结果行
        for row in results:
            print(row)

        # 打印表格底部
        print("╰" + "─" * 5 + "┴" + "─" * max_title_length + "┴" + "─" * 10 + "┴" + "─" * max_curriculum_length + "┴" + "─" * 6 + "┴" + "─" * 8 + "┴" + "─" * 8 + "┴" + "─" * 8 + "┴" + "─" * 8 + "┴" + "─" * 8 + "┴" + "─" * 8 + "┴" + "─" * 8 + "┴" + "─" * 8 + "╯")
        
        # 计算和打印统计数据
        for metric, values in all_metrics.items():
            if values:  # 只有在存在数据的情况下才进行计算
                range_value = np.max(values) - np.min(values)
                mean_value = np.mean(values)
                std_dev_value = np.std(values)

                print(f"{metric} \n \t\t 极差: {range_value:.4f}, 平均值: {mean_value:.4f}, 标准差: {std_dev_value:.4f}")

    else:
        print(f"没有找到作者 {target_author} 的论文。")

# 程序入口
if __name__ == "__main__":
    main()
    print("程序运行完毕！共找到", target_author, "的论文", processed_papers_count, "篇。")

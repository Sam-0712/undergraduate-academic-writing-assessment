# Introduction
This is a natural language processing project that aims to extract features from academic essays. The features are mainly based on the content of the essays, such as the length of each sentence, the number of clauses, the coherence of the sentences, and the consistency of the essay. The project also includes a part of topic modeling analysis, which is based on the LDA algorithm. The project is implemented in Python and uses several libraries, including jieba, snownlp, and pyLDAvis.

The project is divided into several parts:
1. Data preprocessing: The essays are loaded from the specified directory and preprocessed, including tokenization, stop words removal, and sentence splitting. The preprocessed data is stored in a dictionary, where each key is the ID of the essay, and the value is a dictionary containing the preprocessed information.
2. Feature extraction: The preprocessed data is used to extract features, including the average sentence length, the number of clauses, the coherence of the sentences, and the consistency of the essay. The features are stored in the preprocessed data dictionary.
3. Topic modeling analysis: The preprocessed data is used to perform topic modeling analysis, which is based on the LDA algorithm. The analysis results are stored in the preprocessed data dictionary.
4. Visualization: The preprocessed data is used to generate visualizations, including the LDA visualization and the word cloud. The visualizations are saved in the specified directory.

**WARNING**: Since I am majoring in Chinese Language and Literature, I am a beginner in coding and may not be able to provide a high-quality code. That is to say, there may be many errors in the code and the documentation. If you find any errors or have any suggestions, please feel free to contact me.

# Details
## Data preprocessing
The essays are loaded from the specified directory and preprocessed, including tokenization, stop words removal, and sentence splitting. The preprocessed data is stored in a dictionary, where each key is the ID of the essay, and the value is a dictionary containing the preprocessed information.

The tokenization is done using the specified tokenization model, which can be 'jieba', 'hanlp', or'snownlp'. The stop words are removed using the specified stop words file. The sentence splitting is done using regular expressions.

## Feature extraction
The preprocessed data is used to extract features, including the average sentence length, the number of clauses, the coherence of the sentences, and the consistency of the essay. The features are stored in the preprocessed data dictionary.

The average sentence length is calculated by dividing the total number of words by the number of sentences. The number of clauses is calculated by counting the number of commas and semicolons in the text. The coherence of the sentences is calculated by counting the number of coherence keywords in the text. The consistency of the essay is calculated by using the T-Re algorithm.

## Topic modeling analysis
to be done...

## Visualization
to be done...

# Future work
There are several directions for future work:
1. Improve the T-Re algorithm. Currently, the algorithm is based on the assumption that the essay is composed of a series of sentences, and the coherence of the sentences is determined by the number of coherence keywords in the text. However, this assumption may not be valid in some cases.
2. Use more advanced topic modeling techniques, such as HDP, LDA-C, or LDA-M.
3. Use more advanced feature extraction techniques, such as sentiment analysis, named entity recognition, or dependency parsing.
4. Use more advanced visualization techniques, such as network visualization or graph visualization.
5. Use more advanced algorithms for text classification, such as deep learning or support vector machines.
6. Use more advanced algorithms for text clustering, such as k-means or DBSCAN.
7. Use more advanced algorithms for text summarization, such as extractive or abstractive summarization.

The project is still under development and will be updated frequently. If you have any questions or suggestions, please feel free to contact me.

# Acknowledgements
Those who provided their masterpieces for this project are listed below:
- Yuanwen Su (Chinese Language and Literature)
- Zexin He (Mathematics and Applied Mathematics)
- Junhan Zhou (Chemistry)
- Yaohui Xie (Chemistry)

I would like to thank my teacher **Mr.Liu** for his guidance and support. And last but not least, I would like to thank the open-source libraries used in this project.
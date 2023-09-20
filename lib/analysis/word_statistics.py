from typing import Literal
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from spacy.tokens import Doc, Span
from scipy.sparse import csr_matrix, coo_matrix, dok_matrix
import networkx as nx
from pyvis.network import Network
from lib.simple_docs import SimpleDocs, SimpleDoc, SimpleSentence, SimpleWord

class WordFrequency:
    def __init__(self, frequency_matrix:csr_matrix, vocab: np.ndarray, d_word_to_pos: dict[str, str] = None):
        self.frequency_matrix: csr_matrix = frequency_matrix
        self.vocab: np.ndarray = vocab
        self.d_frequency: dict[str, int] = {word:freq for word, freq in zip(vocab, self.frequency_matrix.sum(axis = 0).tolist()[0])}
        self.sorted_tuple: list[tuple[str, int]] = sorted(self.d_frequency.items(), key=lambda x: x[1], reverse=True)
        self.d_word_to_pos: dict[str, str] = d_word_to_pos

    def frequent_words(self, n_words: int = 1000) -> list[str]:
        """上位何単語を返すか

        Args:
            n_words (int, optional): 上位何単語を返すか. Defaults to 1000.

        Returns:
            list[str]: 上位n_wordsの単語
        """
        return [word for word, freq in self.sorted_tuple[:n_words]]

    def to_dataframe(self, n_words = 1000) -> pd.DataFrame:
        """データフレームに変換する。単語数が多いとメモリが足りなくなるので注意。

        Args:
            n_words (int, optional): 上位何単語を返すか. Defaults to 1000.

        Returns:
            pd.DataFrame: カラムが単語、行が文書で出現数を格納するデータフレーム
        """
        frequent_words = self.frequent_words(n_words)
        ivocabs_filtered = [i for i, v in enumerate(self.vocab) if v in frequent_words]
        return pd.DataFrame(self.frequency_matrix[:, ivocabs_filtered].toarray(), columns = self.vocab[ivocabs_filtered])

    def get_frequencies(self, words: list[str], sorted: bool = True)->list[str, int]|list[int]:
        """単語の出現頻度を返す

        Args:
            words (list[str]): 単語のリスト
            sorted (bool, optional): 出現頻度の降順で返すか. Defaults to True.

        Returns:
            list[str, int]|list[int]: 単語と出現頻度のタプルのリスト。sort Falseの場合は頻度のみ。
        """
        if sorted:
            return sorted([(word, self.d_frequency[word]) for word in words], key=lambda x: x[1], reverse=True)
        else:
            return [self.d_frequency[word] for word in words]

class TfIdf:
    def __init__(self, vector_matrix:csr_matrix, vocab: np.ndarray, d_word_to_pos: dict[str, str] = None):
        self.vector_matrix: csr_matrix = vector_matrix
        self.vocab: np.ndarray = vocab
        self.d_vector: dict[str, int] = {word:freq for word, freq in zip(vocab, self.vector_matrix.sum(axis = 0).tolist()[0])}
        self.sorted_tuple: list[tuple[str, int]] = sorted(self.d_vector.items(), key=lambda x: x[1], reverse=True)
        self.d_word_to_pos: dict[str, str] = d_word_to_pos

class WordCoOccurrence:
    def __init__(self, 
                 co_occurrence_matrix: dok_matrix, 
                 words: np.ndarray,
                 frequencies: np.ndarray,
                 d_word_to_pos: dict[str, str] = None,
                 document_length: int = None,
                 ):
        self.co_occurrence_matrix: dok_matrix = co_occurrence_matrix
        self.words: np.ndarray = words
        self.frequencies: np.ndarray = frequencies
        self.d_word_to_pos: dict[str, str] = d_word_to_pos
        self.document_length: int = document_length

    def create_network(self, 
                       n_edge: int = 100,
                       min_co_occurrence: float|int = 0.01,
                       ) -> Network:
        """共起頻度のネットワークを作成する

        Args:
            min_co_occurrence (float, optional): 文書の数に対して共起頻度がどれだけあるかの割合の最小値. または共起頻度の絶対数。 Defaults to 0.01.
            n_edge (int, optional): 作成するエッジの最大数. Defaults to 100.

        Returns:
            Network: pyvis の Network インスタンス
        """
        co_occurence_matrix_with_noise = self.co_occurrence_matrix.astype(np.float32).copy()
        # co_occurence_matrix の 0 ではない要素に 1.e-3 以下の大きさのランダムな数値を加える
        # これによって n_edge で指定した数にかなり近い数のエッジが作成されるようになる
        co_occurence_matrix_with_noise[co_occurence_matrix_with_noise.nonzero()] += np.random.rand(len(co_occurence_matrix_with_noise.nonzero()[0])) * 10.e-3
        max_frequency = max(self.frequencies)
        max_co_occurrence = max(co_occurence_matrix_with_noise.values())
        min_co_occurrence_by_number_of_edge = sorted(co_occurence_matrix_with_noise.values(), reverse=True)[:n_edge][-1]
        if isinstance(min_co_occurrence, int):
            min_co_occurrence_by_hand = min_co_occurrence
        else:
            min_co_occurrence_by_hand = min_co_occurrence * self.document_length
        min_co_occurence = max(min_co_occurrence_by_hand, min_co_occurrence_by_number_of_edge) / max_co_occurrence

        G = nx.Graph()
        nodes_to_show = set()
        edges_to_show = list()
        for (i, j), freq in co_occurence_matrix_with_noise.items():
            weight = freq/max_co_occurrence
            if weight >= min_co_occurence:
                edges_to_show.append((self.words[i], self.words[j], weight))
                nodes_to_show.add(self.words[i])
                nodes_to_show.add(self.words[j])

        for i, word in enumerate(self.words):
            if word in nodes_to_show:
                G.add_node(
                    word, 
                    size = self.frequencies[i]/max_frequency,
                    weight = self.frequencies[i]/max_frequency,
                    )
        for word_i, word_j, weight in edges_to_show:
            G.add_edge(word_i, word_j, weight=weight)

        pyvis_net = Network('600px', '900px')
        d_pos_to_color = {
            "VERB": "#4e79a7",
            "NOUN": "#f28e2b",
            "AUX": "#e15759",
            "ADV": "#e15759",
            "ADJ": "#e15759",
        }
        color_other = "#bab0ac"

        for node, attrs in G.nodes(data=True):            
            color = d_pos_to_color.get(self.d_word_to_pos[node], color_other)
            pyvis_net.add_node(
                node,
                title=node,
                size=30 * attrs['weight'],
                color=color,
                )
        # color_edge = "#a0cbe8a0"
        # color_edge = "#57606ca0"
        # color_edge = "#75b7b2a0"
        # color_edge = "#459792a0"
        # color_edge = "#49914da0"
        # color_edge = "#91215d80"
        color_edge = "#41a1ada0"
        for node1, node2, attrs in G.edges(data=True):
            pyvis_net.add_edge(node1, node2, width=12 * attrs['weight'],
                               color = color_edge)
        # pyvis_net.toggle_physics(False)

        pyvis_net.barnes_hut(
            gravity=-4000,
            central_gravity=2.8,
            spring_length=10.0,
            spring_strength=0.01,
            damping=1.4,
            overlap=0.6)
        if pyvis_net.nodes == []:
            pyvis_net.add_node("表示できる共起表現がありません。\n\n表示する対象の条件を緩めてください", color = "#ffffff00")
        # pyvis_net.save_graph("data/word_co_occurrence.html")
        return pyvis_net

class WordStatistics:
    """単語の頻度や共起頻度を保持・計算するクラス

    Args:
        docs (SimpleDocs): 文書のリスト

    Attributes:
        docs (SimpleDocs): 入力された文書のリスト
        filtered_docs (SimpleDocs): 単語に分割された文書のリスト
        delimeted_docs (list[str]): delimiterで連結された文書のリスト
        word_frequency (WordFrequency): 単語の出現頻度を保持するクラス

    Methods:
        count_words: 単語の出現回数を数える
        get_vocabularies: 頻度の高い上位の語彙を返す
        calculate_co_occurrence: 共起頻度を計算する
        calculate_tf_idf: TF-IDFを各グループわけごとに計算する
    """

    def __init__(
            self, 
            docs: SimpleDocs,
            min_document_frequency: float = 0.02,
            max_document_frequency: float = 0.98,
            max_vocab_size: int = 20000,
            ):
        self.docs = docs
        self.min_document_frequency = min_document_frequency
        self.max_document_frequency = max_document_frequency
        self.max_vocab_size = max_vocab_size

        # self.dict_labels = dict_labels
        self._filtered_docs = None
        self._delimeted_docs = None
        self._word_frequency = None
        # self.word_frequency = self.count_words(self.filtered_docs)
        # self.dict_tf_idf = self.calculate_tf_idf(self.filter_docs, self.dict_labels)
        self.d_word_to_pos = self._create_word_to_pos_dict(self.docs)
        

    @property
    def delimeted_docs(self) -> list[str]:
        """delimiterで連結された文書のリスト

        Returns:
            list[str]: delimiterで連結された文書のリスト
        """
        if self._delimeted_docs is None:
            self._delimeted_docs = self.merge_by_delimiter(self._filtered_docs, self.delimiter)
        return self._delimeted_docs

    @property
    def word_frequency(self) -> WordFrequency:
        """単語の出現頻度を保持するクラス

        Returns:
            WordFrequency: 単語の出現頻度を保持するクラス
        """
        if self._word_frequency is None:
            self._word_frequency = self.count_words(self.filtered_docs)
        return self._word_frequency

    def get_vocabularies(self, n_words:int) -> list[str]:
        """分析対象の単語のリスト

        Args:
            n_words (int): Frequency 上位何単語を出力するか
        Returns:
            list[str]: 分析対象の単語のリスト
        """
        if self.word_frequency:
            return self.word_frequency.frequent_words(n_words=n_words)
        else:
            return None

    def _create_word_to_pos_dict(self, docs: SimpleDocs) -> dict[str, str]:
        """単語と品詞の対応を辞書で作成する

        Args:
            docs (SimpleDocs): 単語に分割された文書のリスト

        Returns:
            dict[str, list[str]]: 単語と品詞の対応を辞書で作成する
        """
        d_word_to_pos = {}
        for doc in docs:
            for sentence in doc:
                for word in sentence:
                    if word.lemma not in d_word_to_pos:
                        d_word_to_pos[word.lemma] = [word.pos]
                    else:
                        d_word_to_pos[word.lemma].append(word.pos)
        
        # 各単語について、多数決で品詞を決める
        for word, pos_list in d_word_to_pos.items():
            d_word_to_pos[word] = max(set(pos_list), key=pos_list.count)
        return d_word_to_pos

    @property
    def filtered_docs(self) -> SimpleDocs:
        """文書中の重要な単語を抽出しリストにする

        Notes:
            - pos_relevant の 品詞の単語をlemma で抽出する
                - ただし、AUXは lemma が 「ない」 となるもののみを抽出する
                - ただし、dep_ が compound のPROPN, NOUNの場合は次の語と連結して抽出する
                - ただし、デフォルトのstopwordは除く

        Returns:
            list[list[str]]: 各文書を抽出した重要単語のリストにしたリスト
        """
        if self._filtered_docs is None:
            docs: SimpleDocs = self.docs
            pos_relevant = ["VERB", "PROPN", "NOUN", "AUX", "ADV", "ADJ"]
            # see [spacyの品詞ラベル](https://yu-nix.com/archives/spacy-pos-list/)
            stop_words_j = ["ます", "てる", "事", "です", "思う", "他", ]
            def is_relevant(word: SimpleWord) -> SimpleWord|None:
                """重要な単語かどうか"""
                if word.lemma in stop_words_j:
                    return False
                elif word.pos in pos_relevant and not word.is_stop:
                    if word.pos != "AUX" or word.lemma == "ない":
                        return True
                    else:
                        return False
                elif word.ent_type:
                    # 固有表現でretokenizeされたものはそのまま追加する
                    return True
                else:
                    False

            docs_filtered = []
            for doc in docs:
                sentences = []
                for sentence in doc:
                    words_filtered = [word for word in sentence if is_relevant(word)]
                    if words_filtered == []:
                        words_filtered = [
                            SimpleWord(
                                text = "",
                                lemma = "",
                                is_stop = True,
                                pos = "NOUN",
                                vector = np.zeros(768),
                                ent_type = None)]
                    sentences.append(SimpleSentence(words = words_filtered))
                docs_filtered.append(SimpleDoc(sentences = sentences))
            self._filtered_docs = docs_filtered
        return self._filtered_docs
    
    @property
    def delimiter(self) -> str:
        """文書をdelimiterで連結する際のdelimiterを返す
        
        Notes:
        - 0x1Dをテキストの区切り文字として使う。
        - 元テキストに含まれていても TextData の text attribute に入る際には削除される。
        - 0x1DはUnicodeのGS（Group Separator）に対応する文字で、テキストに含まれることはほぼないと思われる。

        Returns:
            str: delimiter
        """
        return chr(0x1D)


    @staticmethod
    def merge_by_delimiter(docs: SimpleDocs, delimiter: str) -> list[str]:
        """文書をdelimiterで連結する

        Args:
            docs (SimpleDocs): 単語に分割された文書のリスト

        Returns:
            list[str]: delimiterで連結された文書のリスト
        """
        delimeted_docs = list()
        for filtered_doc in docs:
            delimeted_sentences = list()
            for filtered_sentence in filtered_doc:
                delimeted_sentences.append(delimiter.join([filtered_word.lemma for filtered_word in filtered_sentence]))
            delimeted_docs.append(delimiter.join(delimeted_sentences) + delimiter)
        return delimeted_docs
    
    @property
    def token_pattern(self) -> str:
        """token_patternを返す

        Returns:
            str: token_pattern
        """
        return f'(?u)([^{self.delimiter}]+?){self.delimiter}'

    def count_words(self, 
                    filtered_docs: SimpleDocs = None,
                    min_document_frequency = None,
                    max_document_frequency = None,
                    max_vocab_size = None,
                    ) -> WordFrequency:
        """単語の出現回数を数える

        Args:
            filtered_docs (SimpleDocs): 重要な単語に絞り込まれた文書のリスト
            min_document_frequency (float, optional): 出現頻度の最小値.
            max_document_frequency (float, optional): 出現頻度の最大値.
            max_vocab_size (int, optional): 出力する単語種類数の最大値.

        Returns:
            pd.DataFrame: カラムが単語、行が文書で出現数を格納するデータフレーム
        """
        if filtered_docs is not None:
            delimeted_docs = self.merge_by_delimiter(filtered_docs, self.delimiter)
        else:
            delimeted_docs = self.delimeted_docs

        min_document_frequency = min_document_frequency or self.min_document_frequency
        max_document_frequency = max_document_frequency or self.max_document_frequency
        max_vocab_size = max_vocab_size or self.max_vocab_size

        # docs = [delimiter.join([delimiter.join(sentence) for sentence in doc]) + delimiter for doc in filtered_docs]
        # docs = [delimiter.join(sent) + delimiter for doc in filtered_docs for sent in doc]
        vectorizer_count = CountVectorizer(
            token_pattern=self.token_pattern,
            lowercase=False,
            max_features=max_vocab_size,
            min_df = min_document_frequency,
            max_df = max_document_frequency,
            binary = True, # 1 frequency for each document
            )
        try:
            frequency_matrix = vectorizer_count.fit_transform(delimeted_docs)
            vocab = vectorizer_count.get_feature_names_out()
            return WordFrequency(frequency_matrix, vocab, d_word_to_pos = self.d_word_to_pos)
        except Exception as e:
            print(f"word_statistics error: {e}")
            return None

    def calculate_co_occurrence(self, n_words: int = 20000) -> WordCoOccurrence:
        """共起頻度を計算する

        Returns:
            WordCoOccurrence: 共起頻度を保持するオブジェクト
        """
        word_frequency = self.word_frequency
        if word_frequency is None:
            return None
        
        vocabs_target = self.get_vocabularies(n_words = n_words)
        d_vocab_index = {word: i for i, word in enumerate(vocabs_target)}
        co_occurence_matrix = dok_matrix((len(vocabs_target), len(vocabs_target)), dtype=np.int32)
        # vocabs_target に入っている単語のみを対象として、sentenceごとに共起頻度を coo matrix で計算する

        # sentence-level での共起頻度を計算する
        for doc in self.filtered_docs:
            for sentence in doc:
                sentence_target = list(set([word.lemma for word in sentence if word.lemma in vocabs_target]))
                for i, word1 in enumerate(sentence_target):
                    for j, word2 in enumerate(sentence_target):
                        if i != j:
                            co_occurence_matrix[d_vocab_index[word1], d_vocab_index[word2]] += 1

        # doc-level での共起頻度を計算する
        # for doc in filtered_docs:
        #     doc_target = list(set([word.lemma for sentence in doc for word in sentence if word.lemma in vocabs_target]))
        #     for i, word1 in enumerate(doc_target):
        #         for j, word2 in enumerate(doc_target):
        #             if i != j:
        #                 co_occurence_matrix[d_vocab_index[word1], d_vocab_index[word2]] += 1

        wco =  WordCoOccurrence(
            co_occurence_matrix, 
            vocabs_target, 
            word_frequency.get_frequencies(vocabs_target, sorted=False),
            d_word_to_pos = self.d_word_to_pos,
            document_length = len(self.docs),
            )
        return wco
    
    def calculate_tf_idf(self, filtered_docs: SimpleDocs = None,
                         min_document_frequency: float = 0.0,
                         max_document_frequency: float = 1.00,
                         n_words: int = 3000,
                        # TODO: vocab を min_df, max_dfからきたものにする
                         labels: list = None,
                         )->TfIdf:
        """TF-IDFを計算する

        Args:
            filtered_docs (list[list[str]]): 各docの各sentenceの重要単語のリスト
            labels (list, optional): 各文書のラベル. Defaults to None.

        Returns:
            pd.DataFrame: カラムが単語、行が文書でTF-IDFを格納するデータフレーム
        """
        if filtered_docs is not None:
            delimeted_docs = self.merge_by_delimiter(filtered_docs, self.delimiter)
        else:
            delimeted_docs = self.delimeted_docs

        # labels の 値ごとにdelimeted_docsを分割する
        if labels is None:
            labels = [0] * len(delimeted_docs)
        d_label_to_delimeted_docs = {}
        for label, delimeted_doc in zip(labels, delimeted_docs):
            if label not in d_label_to_delimeted_docs:
                d_label_to_delimeted_docs[label] = delimeted_doc
            else:
                d_label_to_delimeted_docs[label] = d_label_to_delimeted_docs[label] + delimeted_doc

        labels_unique = sorted(list(set(labels)))
        vectorizer_tfidf = TfidfVectorizer(
            token_pattern=self.token_pattern,
            lowercase=False,
            vocabulary=self.get_vocabularies(n_words = n_words),
            min_df = min_document_frequency,
            max_df = max_document_frequency,
        )
        
        corpus = [d_label_to_delimeted_docs[label] for label in labels_unique]
        try:
            tfidf_matrix = vectorizer_tfidf.fit_transform(corpus)
            vocab = vectorizer_tfidf.get_feature_names_out()
            return TfIdf(tfidf_matrix, vocab, d_word_to_pos = self.d_word_to_pos)
        except Exception as e:
            print(f"word_statistics tf-idf error: {e}")
            return None
        

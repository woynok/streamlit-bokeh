from __future__ import annotations
import re
import numpy as np
import pandas as pd
import pickle
from . import AuxiliaryDataColumns
from lib.simple_docs import SimpleDocs

class TextData:
    """文書のテキストデータを保持するクラス

    事前にやっておきたい重い処理はこのクラスで行う。

    Args:
        texts (list[str]): 文書のテキストのリスト
        aux_data (AuxiliaryDataColumns): 文書のテキスト以外のデータを保持する
        preprocess_for_sentence (function, optional): 文書の前処理を行う関数. Defaults to None.
        embeddings (np.ndarray, optional): 文書の埋め込みベクトル. Defaults to None.
        docs (list[str], optional): 文書のlist. Defaults to None.
        configuration (dict, optional): 設定. Defaults to None.
    
    Attributes:
        texts (list[str]): 文書のテキストのリスト
        aux_data (AuxiliaryDataColumns): 文書のテキスト以外のデータを保持する
        preprocess_for_sentence (function): 文書の前処理を行う関数
        embeddings (np.ndarray): 文書の埋め込みベクトル
        docs (list[str]|SimpleDocs): 文書のlist。tokenize_docs() するとSimpleDocsになる
        configuration (dict): 設定
    
    Methods:
        calculate_embeddings: 文書の埋め込みベクトルを計算する
        tokenize_docs: 文書を単語に分割する
        from_dataframe: DataFrameからTextDataを作成する
        save: 自分自身をpickleで保存する
        load: pickleで保存されたインスタンスを読み込む

    """

    MAX_DOCUMENT_LENGTH_TO_PROCESS = 2000  # 処理する文書の文字数を2000文字までにリミットする
    MAX_DOCUMENT_LENGTH_TO_STORE = 1200  # 保持する文書の文字数を1200文字までにリミットする
    MAX_DOCUMENT_SIZE = 3000  # 3000文書までにリミットする。aux_dataも同じサイズにリミットする。

    def __init__(
        self,
        texts: list[str],
        aux_data: AuxiliaryDataColumns,
        preprocess_for_sentence=None,
        embeddings: np.ndarray = None,
        docs: list[str]|SimpleDocs = None,
        configuration: dict = None,
        description_short: str = "",
        description_detail: str = "",
        _meta = None,
    ):
        self.preprocess_for_sentence = preprocess_for_sentence
        self.texts = texts
        self.aux_data = aux_data
        self._embeddings = embeddings
        self._docs = docs
        self._meta = _meta
        self.description_short = description_short
        self.description_detail = description_detail

        if configuration:
            self.configuration = configuration
        else:
            self.configuration = {
                "pickle_protocol": 4,
                "tokenizer": "ja_ginza_electra_with_vector",
                "embedding_model": "pkshatech/GLuCoSE-base-ja",
            }

    @property
    def texts(self) -> list[str]:
        return self._texts

    @texts.setter
    def texts(self, texts_given: list[str]):
        """入力されたテキストを前処理して保持する

        Args:
            texts_given (list[str]): 入力されたテキストのリスト

            Notes:
                - 連続する空白を1つにする
                - 全角スペース、タブ、改行を半角スペースに置換する
                - 0x1Dを半角スペースに置換する。0x1Dを区切り文字として使うため必須。
                - 先頭から MAX_DOCUMENT_LENGTH_TO_PROCESS 文字までを処理し、最終的に MAX_DOCUMENT_LENGTH_TO_STORE 文字までを残す
                - 文書数を MAX_DOCUMENT_SIZE までにリミットする

            Returns:
                list[str]: 前処理されたテキストのリスト
        """

        def preprocess_default(text: str) -> str:
            if not isinstance(text, str):
                return ""
            text = (
                text[: self.__class__.MAX_DOCUMENT_LENGTH_TO_PROCESS]
                .replace("　", " ")
                .replace(chr(0x1D), " ")
                .replace("\t", " ")
                .replace("\r", " ")
            )
            # text 内の 連続する空白を1つにする
            text = re.sub(r" +", " ", text)
            return text.strip()[: self.__class__.MAX_DOCUMENT_LENGTH_TO_STORE]

        if not self.preprocess_for_sentence:
            preprocess = preprocess_default
        else:
            preprocess = self.preprocess_for_sentence
        self._texts = [
            preprocess(text)
            for itext, text in enumerate(texts_given)
            if itext < self.__class__.MAX_DOCUMENT_SIZE
        ]

    @property
    def aux_data(self) -> AuxiliaryDataColumns:
        return self._aux_data

    @aux_data.setter
    def aux_data(self, aux_data_given: AuxiliaryDataColumns):
        # MAX_DOCUMENT_SIZE にリミットする
        for column in aux_data_given.columns:
            column.records = column.records[: self.__class__.MAX_DOCUMENT_SIZE]
        self._aux_data = aux_data_given

    @property
    def embeddings(self) -> np.ndarray:
        return self._embeddings

    @property
    def docs(self) -> SimpleDocs|list[str]:
        return self._docs

    def calculate_embeddings(self):
        """文書の埋め込みベクトルを計算する

        Returns:
            np.ndarray: 文書の埋め込みベクトル
        """
        import torch
        from sentence_transformers import SentenceTransformer
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        model = SentenceTransformer(
            self.configuration["embedding_model"], device=device
        )
        self._embeddings = model.encode(self.texts)

    def tokenize_docs(self, reduce_vector_dimension = True):
        """文書を単語に分割する

        Returns:
            list[str]: 文書のlist
        """
        import spacy
        from spacy.language import Language
        from lib.simple_docs import SimpleDocs
        if self.configuration["tokenizer"] == "ja_ginza_electra_with_vector":
            from .trf_vector import TrfVectors
            nlp = spacy.load("ja_ginza_electra")
            @Language.factory('trf_vector', assigns=["doc._.doc_vector"])
            def create_trf_vector_hook(nlp, name):
                return TrfVectors(nlp)
            # vector をつけるとかなり重くなる。random projection で次元削減
            nlp.add_pipe("trf_vector", last=True)
        else:
            nlp = spacy.load(self.configuration["tokenizer"])
        
        docs = [nlp(text) for text in self.texts]
        simple_docs = SimpleDocs.from_docs(docs)
        if reduce_vector_dimension:
            vectors = []
            for doc in simple_docs:
                for sentence in doc:
                    for word in sentence:
                        vectors.append(word.vector)
            vectors = np.array(vectors)
            from sklearn.random_projection import GaussianRandomProjection
            reducer = GaussianRandomProjection(n_components = 768//8)
            vectors = reducer.fit_transform(vectors)
            ivector = 0
            for doc in simple_docs:
                for sentence in doc:
                    for word in sentence:
                        word.vector = vectors[ivector]
                        ivector += 1
        self._docs = simple_docs

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, text_column_name: str):
        """DataFrameからTextDataを作成する

        Args:
            df (pd.DataFrame): テキストデータを含むDataFrame
            text_column_name (str): テキストデータのカラム名

        Returns:
            TextData: TextData
        """        
        aux_data = AuxiliaryDataColumns.from_dataframe(df, exclude_columns=[text_column_name])
        texts = df[text_column_name].tolist()
        return cls(texts, aux_data)

    def save(self, filename: str = "data/text_data.pkl"):
        """自分自身をpickleで保存する"""
        with open(filename, "wb") as f:
            pickle.dump(self, f, self.configuration["pickle_protocol"])
    
    @classmethod
    def load(cls, filename: str = "data/text_data.pkl")-> TextData:
        """pickleで保存されたインスタンスを読み込む"""
        with open(filename, "rb") as f:
            return pickle.load(f)
    
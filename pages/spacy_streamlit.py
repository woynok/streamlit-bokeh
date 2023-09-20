# streamlit_app.py
import spacy_streamlit

import itertools
import sys

import numpy as np
import spacy
from spacy.language import Language
from spacy.tokens import Doc

# try:
#     import cupy    # spaCyでGPUが使用可能な場合はcuPyをインポートする必要がある
# except ImportError:
#     pass

class TrfVectors:
    def __init__(self, name):
        self.name = name

    def __call__(self, doc):
        if not Doc.has_extension('doc_vector'):    # doc._.doc_vectorというattributeを追加
            Doc.set_extension('doc_vector', default=None)
        doc._.doc_vector = self._trf_vectors_getter(doc)    # docを得る際に単語ベクトルを計算する

        # token.vector, span.vector, doc.vectorの定義
        doc.user_token_hooks["vector"] = self.vector
        doc.user_span_hooks["vector"] = self.vector
        doc.user_hooks["vector"] = self.vector

        # token.has_vector, span.has_vector, doc.has_vectorの定義
        doc.user_token_hooks["has_vector"] = self.has_vector
        doc.user_span_hooks["has_vector"] = self.has_vector
        doc.user_hooks["has_vector"] = self.has_vector

        # token.similarity, span.similarity, doc.similarityの定義
        doc.user_token_hooks["similarity"] = self.similarity
        doc.user_span_hooks["similarity"] = self.similarity
        doc.user_hooks["similarity"] = self.similarity

        return doc

    def _trf_vectors_getter(self, doc):
        vectors = []
        tensors = doc._.trf_data.tensors[0]
        tensors = np.reshape(tensors, (tensors.shape[0] * tensors.shape[1], -1))
        for idx, token in enumerate(doc):
            token_idxes = list(itertools.chain.from_iterable(token.doc._.trf_data.align[idx].data))
            if token_idxes:
                vectors.append(np.mean(tensors[token_idxes], axis=0))
            else:
                vectors.append(np.zeros(768))    # 入力トークンが未知語だった場合はゼロベクトルとする
        return np.array(vectors)

    def vector(self, item):
        if isinstance(item, spacy.tokens.Token):
            idx = item.i
            return item.doc._.doc_vector[idx]
        else:
            return np.mean([token.vector for token in item], axis=0)

    def has_vector(self, item):
        return True

    def similarity(self, obj1, obj2):
        v1 = obj1.vector
        v2 = obj2.vector
        try:
            return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))    # コサイン類似度
        except ZeroDivisionError:
            return 0.0


@Language.factory('trf_vector', assigns=["doc._.doc_vector"])
def create_trf_vector_hook(nlp, name):
    return TrfVectors(nlp)


models = ["ja_ginza_electra", "ja_ginza"]
default_text = "Sundar Pichai is the CEO of Google."
spacy_streamlit.visualize(models, default_text)

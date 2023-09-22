from __future__ import annotations
from typing import Iterator
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from spacy.language import Language
from spacy.tokens import Token, Span, Doc

class SimpleWord:
    def __init__(self, text, lemma, is_stop, pos, vector, ent_type):
        self.text: str = text
        self.lemma: str = lemma
        self.is_stop: bool = is_stop
        self.pos: str = pos
        self.vector: np.ndarray = vector
        self.ent_type: str = ent_type
    
    @staticmethod
    def from_span(span:Span|Token)->SimpleWord:
        # compound な token による spanが入ってきているとする
        text = span.text
        lemma = span.lemma_
        is_stop = False
        pos = span[-1].pos_
        vector = span.vector
        ent_type = span[-1].ent_type_
        return SimpleWord(text, lemma, is_stop, pos, vector, ent_type)
    
    def __repr__(self) -> str:
        return f"<SimpleWord: {self.text}>"
    
    def __len__(self)->int:
        return len(self.text)
    
    def to_dict(self, without_vector = True)->dict:
        if without_vector:
            return {
                "text": self.text,
                "lemma": self.lemma,
                "is_stop": self.is_stop,
                "pos": self.pos,
                "ent_type": self.ent_type,
            }
        else:
            return {
                "text": self.text,
                "lemma": self.lemma,
                "is_stop": self.is_stop,
                "pos": self.pos,
                "vector": self.vector,
                "ent_type": self.ent_type,
            }
    
    @staticmethod
    def from_dict(d:dict)->SimpleWord:
        return SimpleWord(
            text = d["text"],
            lemma = d["lemma"],
            is_stop = d["is_stop"],
            pos = d["pos"],
            vector = d.get("vector", None),
            ent_type = d["ent_type"],
        )
    
    def similarity(self, other:SimpleWord|str|np.ndarray, model: SentenceTransformer)->float:
        v1 = model.encode(self.text)
        if isinstance(other, str):
            v2 = model.encode(other)
        elif isinstance(other, np.ndarray):
            v2 = other
        else:
            v2 = model.encode(other.text)
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    
    def similarity_ginza(self, other:SimpleWord|str|np.ndarray, nlp: Language = None)->float:
        v1 = self.vector_ginza
        if isinstance(other, str):
            v2 = nlp(other).vector
        elif isinstance(other, np.ndarray):
            v2 = other
        else:
            v2 = other.vector_ginza
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    
    @property
    def vector_ginza(self):
        return self.vector

class SimpleSentence:
    def __init__(self, words:list[SimpleWord]):
        self.words: list[SimpleWord] = words
        self.text: str = "".join([word.text for word in words])

    @staticmethod
    def from_sentence(sentence:Doc)->SimpleSentence:
        words = []
        for itoken in range(len(sentence)):
            token = sentence[itoken]
            if token.dep_ == "compound":
                jtoken = itoken
                is_compound = True
                while is_compound:
                    jtoken += 1
                    try:
                        is_compound = sentence[jtoken].dep_ == "compound"
                    except:
                        is_compound = False
                span = sentence[itoken:jtoken+1]
                words.append(SimpleWord.from_span(span))
                itoken = jtoken
            else:
                text = token.text
                lemma = token.lemma_
                is_stop = token.is_stop
                pos = token.pos_
                vector = token.vector
                ent_type = token.ent_type_
                words.append(SimpleWord(text, lemma, is_stop, pos, vector, ent_type))
        return SimpleSentence(words)
    
    def __repr__(self) -> str:
        return f"<SimpleSentence: {self.text}>"
    
    def __getitem__(self, index:int)->SimpleWord:
        return self.words[index]
    
    def __len__(self)->int:
        return len(self.words)
    
    def __iter__(self)->Iterator[SimpleWord]:
        return iter(self.words)
    
    def similarity(self, other:SimpleSentence|str|np.ndarray, model: SentenceTransformer)->float:
        v1 = model.encode(self.text)
        if isinstance(other, str):
            v2 = model.encode(other)
        elif isinstance(other, np.ndarray):
            v2 = other
        else:
            v2 = model.encode(other.text)
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    
    def similarity_ginza(self, other:SimpleSentence|str|np.ndarray, nlp: Language = None)->float:
        v1 = self.vector_ginza
        if isinstance(other, str):
            v2 = nlp(other).vector
        elif isinstance(other, np.ndarray):
            v2 = other
        else:
            v2 = other.vector_ginza
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    
    @property
    def vector_ginza(self):
        v = np.zeros(self.words[0].vector.shape)
        for word in self.words:
            v += word.vector
        return v / len(self.words)
    
    def to_word_list(self, without_vector = True)->list[dict[str, str|bool|np.ndarray]]:
        return [word.to_dict(without_vector = without_vector) for word in self.words]
    
    @staticmethod
    def from_dict(d:dict)->SimpleSentence:
        return SimpleSentence(d["text"], [SimpleWord.from_dict(word) for word in d["words"]])
            
class SimpleDoc:
    def __init__(self, sentences:list[SimpleSentence]):
        self.sentences = sentences

    @staticmethod
    def from_doc(doc:Doc)->SimpleDoc:
        sentences = []
        for sentence in doc.sents:
            sentences.append(SimpleSentence.from_sentence(sentence))
        return SimpleDoc(sentences)
    
    def similarity(self, other:SimpleDoc|str|np.ndarray, model:SentenceTransformer)->float:
        v1 = model.encode(self.text)
        if isinstance(other, str):
            v2 = model.encode(other)
        elif isinstance(other, np.ndarray):
            v2 = other
        else:
            v2 = model.encode(other.text)
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    
    def similarity_ginza(self, other:SimpleDoc|str|np.ndarray, nlp: Language = None)->float:
        v1 = self.vector_ginza
        if isinstance(other, str):
            v2 = nlp(other).vector
        elif isinstance(other, np.ndarray):
            v2 = other
        else:
            v2 = other.vector_ginza
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    @property
    def text(self)->str:
        return "".join([sentence.text for sentence in self.sentences])
    
    @property
    def vector_ginza(self):
        v = np.zeros(self.sentences[0].vector_ginza.shape)
        for sentence in self.sentences:
            v += sentence.vector_ginza
        return v / len(self.sentences)
    
    def __repr__(self) -> str:
        return f"<SimpleDoc: {self.sentences}>"

    def __getitem__(self, index:int)->SimpleSentence:
        return self.sentences[index]
    
    def __len__(self)->int:
        return len(self.sentences)
    
    def __iter__(self)->Iterator[SimpleSentence]:
        return iter(self.sentences)
    
    def to_sentence_list(self, without_vector=True)->list[list[dict[str, str|bool|np.ndarray]]]:
        return [sentence.to_word_list(without_vector=without_vector) for sentence in self.sentences]
    
    @staticmethod
    def from_dict(d:dict)->SimpleDoc:
        return SimpleDoc([SimpleSentence.from_dict(sentence) for sentence in d["sentences"]])
        
class SimpleDocs:
    def __init__(self, docs:list[SimpleDoc]):
        self.docs = docs
    
    @staticmethod
    def from_docs(docs:list[Doc])->SimpleDocs:
        simple_docs = SimpleDocs([SimpleDoc.from_doc(doc) for doc in docs])
        return simple_docs
    
    def __getitem__(self, index:int)->SimpleDoc:
        return self.docs[index]
    
    def __len__(self)->int:
        return len(self.docs)
    
    def __iter__(self)->Iterator[SimpleDoc]:
        return iter(self.docs)
    
    def to_dict(self, separate_vectors = True)->dict[str, list[list[dict[str, str|bool|np.ndarray]]] | np.ndarray]:
        if separate_vectors:
            vectors = []
            for doc in self.docs:
                for sentence in doc:
                    for word in sentence:
                        vectors.append(word.vector)
            vectors = np.array(vectors)
            return {
                "docs": [doc.to_sentence_list(without_vector=True) for doc in self.docs],
                "word_vectors": vectors
            }
        else:
            return {
                "docs": [doc.to_sentence_list(without_vector=False) for doc in self.docs],
            }
    
    @staticmethod
    def from_dict(self, dumped_obj:dict[str, list[list[dict[str, str|bool|np.ndarray]]] | np.ndarray])->SimpleDocs:
        if "word_vectors" in dumped_obj.keys():
            # docs and vectors are separately dumped
            vectors = dumped_obj["word_vectors"]
            iword = 0
            docs = []
            for doc in dumped_obj["docs"]:
                sentences = []
                for sentence in doc:
                    words = []
                    for word in sentence:
                        simple_word = SimpleWord.from_dict(word)
                        simple_word.vector = vectors[iword]                        
                        words.append(simple_word)
                        iword += 1
                    sentences.append(SimpleSentence(words))
                docs.append(SimpleDoc(sentences))
            return SimpleDocs(docs)
        else:
            docs = []
            for doc in dumped_obj["docs"]:
                sentences = []
                for sentence in doc:
                    words = []
                    for word in sentence:
                        words.append(SimpleWord.from_dict(word))
                    sentences.append(SimpleSentence(words))
                docs.append(SimpleDoc(sentences))
            return SimpleDocs(docs)

    def to_pickle(self, path:str):
        with open(path, "wb") as f:
            pickle.dump(self, f)
    
    @staticmethod
    def from_pickle(path:str)->SimpleDocs:
        with open(path, "rb") as f:
            return pickle.load(f)

    def __repr__(self) -> str:
        out = "SimpleDocs([\n"
        for doc in self.docs:
            out += f"{doc}, \n"
        out += "])"
        return out
    
    @property
    def texts(self)->list[str]:
        return [doc.text for doc in self.docs]

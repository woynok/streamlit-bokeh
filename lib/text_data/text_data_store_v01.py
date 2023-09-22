from __future__ import annotations
import os
from typing import Literal
from io import BytesIO
import numpy as np
from lib.text_data.auxiliary_data_columns import AuxiliaryDataColumns
from lib.text_data.text_data import TextData
from lib.simple_docs import SimpleDocs, SimpleDoc, SimpleSentence, SimpleWord
import json
import tarfile
from datetime import datetime
from pytz import timezone
import yaml

TEXTDATA_VERSION = "0.1.1"
TZ_TOKYO = timezone("Asia/Tokyo")


class DumpFileFormat:
    """dump する／されたファイルのフォーマットを保持／解決するクラス

    Args:
        base_filepath (str): dump する／されたファイルのベースファイルパス
        tar_filepath (str): dump する／されたファイルのtarファイルパス
        tar_read_mode (str): 読み込みの際にtarfile.open() に渡すパラメータ
        tar_write_mode (str): 書き込みの際にtarfile.open() に渡すパラメータ
        gzip (bool): gzip するかどうか"""

    def __init__(
        self,
        base_filepath: str,
        tar_filepath: str,
        tar_read_mode: str,
        tar_write_mode: str,
        gzip: bool,
        tar_encoding: str = "utf-8",
    ):
        self.base_filepath = base_filepath
        self.tar_filepath = tar_filepath
        self.tar_read_mode = tar_read_mode
        self.tar_write_mode = tar_write_mode
        self.gzip = gzip
        self.tar_encoding = tar_encoding

    @staticmethod
    def from_ambiguous_input(filepath: str, gzip: bool) -> DumpFileFormat:
        """filepathの拡張子を優先しつつgzip引数を考慮して、DumpFileFormatを作成する

        Args:
            filepath (str): dump する／されたファイルのファイルパス
            gzip (bool): gzip するかどうか。ただし、filepathの拡張子が .tds.gz または .tds の場合は無視される
        """
        if filepath.endswith(".tds.gz"):
            gzip = True
            base_filepath = filepath[:-7]
            tar_filepath = filepath
            tar_read_mode = "r:gz"
            tar_write_mode = "w:gz"
        elif filepath.endswith(".tds"):
            gzip = False
            base_filepath = filepath[:-4]
            tar_filepath = filepath
            tar_read_mode = "r"
            tar_write_mode = "w"
        else:
            gzip = gzip
            base_filepath = filepath
            tar_filepath = filepath + ".tds.gz"
            tar_read_mode = "r:gz"
            tar_write_mode = "w:gz"
        return DumpFileFormat(
            base_filepath=base_filepath,
            tar_filepath=tar_filepath,
            tar_read_mode=tar_read_mode,
            tar_write_mode=tar_write_mode,
            gzip=gzip,
        )

    def to_tar_param(self, mode: Literal["read", "write"] = "read") -> dict[str, str]:
        """tarfile.open() に渡すパラメータを返す"""
        if mode == "read":
            return {
                "name": self.tar_filepath,
                "mode": self.tar_read_mode,
                "encoding": self.tar_encoding,
            }
        elif mode == "write":
            return {
                "name": self.tar_filepath,
                "mode": self.tar_write_mode,
                "encoding": self.tar_encoding,
            }
        else:
            raise ValueError(f"Unknown mode: {mode}")


def _get_meta_filepath(base_filepath: str) -> str:
    # version 変更にかかわらずつかう。 override してはならない
    return f"{base_filepath}__td__meta__.yaml"


class DumpFileParameterBase:
    """dump する／されたファイルのメタ情報を保持・取得するクラス

    Args:
        base_filepath (str): dump する／されたファイルのベースファイルパス
        filepaths (dict[str, str]): dump する／されたファイルのファイルパス
        gzip (bool): gzip するかどうか
        version (str): バージョン
        saved_at (str): 保存日時

    Methods:
        from_meta_file: metaファイルを開き、インスタンスを作成する
        to_dict: 辞書に変換する
        create: インスタンスを作成する
        dump: dump する
        _get_meta: metaファイルを取得する
        _create_filepaths: filepathsを作成する
        _check_version: バージョンをチェックする
        _from_meta: metaファイルからインスタンスを作成する
    """

    def __init__(
        self,
        base_filepath: str,
        filepaths: dict[str, str] = None,
        gzip: bool = False,
        version: str = None,
        saved_at: str = None,
    ):
        self.base_filepath = base_filepath
        self._filepaths = filepaths
        self.gzip = gzip
        self.version = version if version else TEXTDATA_VERSION
        self.saved_at = (
            saved_at
            if saved_at
            else datetime.now(tz=TZ_TOKYO).strftime("%Y-%m-%d %H:%M:%S")
        )

    @property
    def filepaths(self):
        return self._filepaths

    @filepaths.setter
    def filepaths(self, filepaths: dict[str, str]):
        if filepaths is not None:
            self._filepaths = filepaths
            return
        filepaths = self._create_filepaths(self.base_filepath)
        self._filepaths = filepaths

    @classmethod
    def _get_meta(
        cls,
        dump_file_format: DumpFileFormat,
    ) -> dict:
        # override してもよい
        meta_filepath = _get_meta_filepath(dump_file_format.base_filepath)
        meta_filename = os.path.basename(meta_filepath)
        with tarfile.open(**dump_file_format.to_tar_param()) as tar:
            meta = tar.extractfile(meta_filename)
            meta = yaml.load(meta, Loader=yaml.SafeLoader)
            tar.close()
        return meta

    @classmethod
    def _create_filepaths(cls, base_filepath) -> dict[str, str]:
        # override してもよいが、"meta" の key は変更しないこと
        # docs のような 大きな辞書はjsonで保存
        return {
            "docs": f"{base_filepath}__td__docs__.json",
            "word_vectors": f"{base_filepath}__td__wvs__.npy",
            "doc_vectors": f"{base_filepath}__td__dvs__.npy",
            "aux_info": f"{base_filepath}__td__aux_info__.yaml",
            "aux_data": f"{base_filepath}__td__aux_data__.npy",
            "text_data_info": f"{base_filepath}__td__td_info__.yaml",
            "meta": f"{_get_meta_filepath(base_filepath)}",
        }

    @classmethod
    def _check_version(cls, meta: dict):
        # override してもよい
        return meta

    @classmethod
    def _from_meta(cls, meta: dict):
        # override してもよい
        return cls(
            base_filepath=meta["base_filepath"],
            filepaths=meta["filepaths"],
            gzip=meta["gzip"],
            version=meta["version"],
            saved_at=meta["saved_at"],
        )

    @classmethod
    def from_meta_file(
        cls,
        dump_file_format: DumpFileFormat,
    ) -> DumpFileParameterBase:
        meta = cls._get_meta(dump_file_format)
        meta = cls._check_version(meta)
        return cls._from_meta(meta)

    def to_dict(self):
        # override してもよい
        return {
            "base_filepath": self.base_filepath,
            "filepaths": self.filepaths,
            "gzip": self.gzip,
            "version": self.version,
            "saved_at": self.saved_at,
        }

    @classmethod
    def create(
        cls,
        dump_file_format: DumpFileFormat,
    ) -> DumpFileParameterBase:
        return cls(
            base_filepath=dump_file_format.base_filepath,
            filepaths=cls._create_filepaths(dump_file_format.base_filepath),
            gzip=dump_file_format.gzip,
            version=TEXTDATA_VERSION,
        )


class DumpFileParameterV01(DumpFileParameterBase):
    version: str = TEXTDATA_VERSION

    def __init__(
        self,
        base_filepath: str,
        filepaths: dict[str, str] = None,
        gzip: bool = False,
        version: str = None,
        saved_at: str = None,
    ):
        super().__init__(
            base_filepath=base_filepath,
            filepaths=filepaths,
            gzip=gzip,
            version=version,
            saved_at=saved_at,
        )

    @classmethod
    def _check_version(cls, meta: dict) -> dict:
        version: str = meta["version"]
        if version[:3] != cls.version[:3]:
            raise ValueError(f"version {meta['version']} is not supported.")
        return meta


class TextDataStoreV01:
    version = TEXTDATA_VERSION
    """TextDataをdumpするクラス
    
    Args:
        version (str): バージョン
        
    Methods:
        load: dumpされたTextDataを読み込む
        dump: TextDataをdumpする
        _from_dict: dictからTextDataを作成する
        _construct_simple_docs: SimpleDocsを作成する
        _from_bytes_to_var: bytesから変数に変換する
        _from_var_to_bytes: 変数からbytesに変換する
    """

    def __init__(
        self,
    ):
        self.version = TEXTDATA_VERSION

    @classmethod
    def load(cls, filepath: str, gzip: bool = False) -> TextData:
        dump_file_format = DumpFileFormat.from_ambiguous_input(filepath, gzip)
        dump_file_parameter = DumpFileParameterV01.from_meta_file(dump_file_format)
        dict_data = dict()
        dict_data["meta"] = dump_file_parameter.to_dict()
        with tarfile.open(**dump_file_format.to_tar_param()) as tar:
            for name, filepath in dump_file_parameter.filepaths.items():
                if name == "meta":
                    continue
                filename = os.path.basename(filepath)
                if filename not in tar.getnames():
                    raise FileNotFoundError(
                        f"{filename} is not found in {dump_file_format.tar_filepath}."
                    )
                dict_data[name] = cls._from_bytes_to_var(
                    name, tar.extractfile(filename)
                )
            tar.close()
        return cls._from_dict(dict_data)

    @classmethod
    def _from_dict(cls, dict_data: dict) -> TextData:
        meta = dict_data["meta"]
        aux_data = AuxiliaryDataColumns.load(
            {
                "info": dict_data["aux_info"],
                "data": dict_data["aux_data"],
            }
        )
        embeddings = dict_data["doc_vectors"]
        docs = cls._construct_simple_docs(
            dumped_doc_list=dict_data["docs"], word_vectors=dict_data["word_vectors"]
        )
        tdinfo = dict_data["text_data_info"]
        return TextData(
            texts=docs.texts,
            aux_data=aux_data,
            embeddings=embeddings,
            docs=docs,
            configuration=tdinfo["configuration"],
            description_short=tdinfo["description_short"],
            description_detail=tdinfo["description_detail"],
            _meta=meta,
        )

    @classmethod
    def _construct_simple_docs(
        cls, dumped_doc_list: list, word_vectors: np.ndarray
    ) -> SimpleDocs:
        docs: list[SimpleDoc] = []
        for doc in dumped_doc_list:
            sentences = []
            for sentence in doc:
                words = []
                for word in sentence:
                    words.append(SimpleWord.from_dict(word))
                sentences.append(SimpleSentence(words))
            docs.append(SimpleDoc(sentences))
        iword = 0
        for doc in docs:
            for sentence in doc:
                for word in sentence:
                    word.vector = word_vectors[iword]
                    iword += 1
        return SimpleDocs(docs)

    @classmethod
    def dump(
        cls,
        text_data: TextData,
        filepath: str,
        gzip: bool = False,
    ):
        dump_file_format = DumpFileFormat.from_ambiguous_input(filepath, gzip)
        dump_file_parameter: DumpFileParameterV01 = DumpFileParameterV01.create(
            dump_file_format,
        )

        dict_data = dict()
        dict_data["meta"] = dump_file_parameter.to_dict()
        dumped_docs = text_data.docs.to_dict()
        dict_data["docs"] = dumped_docs["docs"]
        dict_data["word_vectors"] = dumped_docs["word_vectors"]
        dumped_aux = text_data.aux_data.dump()
        dict_data["aux_info"] = dumped_aux["info"]
        dict_data["aux_data"] = dumped_aux["data"]
        dict_data["doc_vectors"] = text_data.embeddings
        dict_data["text_data_info"] = {
            "configuration": text_data.configuration,
            "description_short": text_data.description_short,
            "description_detail": text_data.description_detail,
        }
        with tarfile.open(**dump_file_format.to_tar_param(mode="write")) as tar:
            for name, content in dict_data.items():
                if name == "meta":
                    continue
                content = cls._from_var_to_bytes(name, content)
                filename_flat = os.path.basename(dump_file_parameter.filepaths[name])
                tarinfo = tarfile.TarInfo(name = filename_flat)
                tarinfo.size = len(content)
                tar.addfile(tarinfo, BytesIO(content))
            # meta を最後に保存する
            content = cls._from_var_to_bytes("meta", dict_data["meta"])
            filename_flat = os.path.basename(dump_file_parameter.filepaths["meta"])
            tarinfo = tarfile.TarInfo(name=filename_flat)
            tarinfo.size = len(content)
            tar.addfile(tarinfo, BytesIO(content))
            tar.close()

    @classmethod
    def _from_bytes_to_var(cls, name: str, content: BytesIO):
        if name in ["docs", "aux_info"]:
            return json.load(content)
        elif name in ["word_vectors", "doc_vectors", "aux_data"]:
            # file in tarfile は np.load() がこけるので、通常の BytesIO に変換する
            bio = BytesIO(content.read())
            return np.load(bio)
        elif name in ["aux_info", "text_data_info", "meta"]:
            return yaml.load(content, Loader=yaml.SafeLoader)
        else:
            raise ValueError(f"Unknown name: {name}")

    @classmethod
    def _from_var_to_bytes(cls, name: str, content) -> bytes:
        if name in ["docs", "aux_info"]:
            return json.dumps(content, ensure_ascii=False).encode("utf-8")
        elif name in ["word_vectors", "doc_vectors", "aux_data"]:
            buffer = BytesIO()
            np.save(buffer, content, allow_pickle=False,)
            return buffer.getvalue()
        elif name in ["aux_info", "text_data_info", "meta"]:
            return yaml.dump(content, encoding="utf-8")
        else:
            raise ValueError(f"Unknown name: {name}")

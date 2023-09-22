from __future__ import annotations
from typing import Iterator
import numpy as np
import pandas as pd

class AuxiliaryDataColumn:
    def __init__(
        self,
        column_name: str,
        records: list|np.ndarray = None,
        is_date: bool = False,
        is_label: bool = False,
        is_train_data: bool = False,
        is_generated: bool = False,
    ):
        self.column_name: str = column_name
        self.records: np.ndarray = records
        self.is_date: bool = is_date
        self.is_label: bool = is_label
        self.is_train_data: bool = is_train_data
        self.is_generated: bool = is_generated
    
    @property
    def records(self) -> np.ndarray:
        return self._records
    
    @records.setter
    def records(self, records_given: list|np.ndarray):
        if isinstance(records_given, list):
            records_given = np.array(records_given)
        self._records = records_given

    def __repr__(self) -> str:
        return f"AuxiliaryDataColumn({self.column_name=}, {self.is_date=}, {self.is_label=}, {self.is_train_data=}, {self.is_generated=})"
    
    def __len__(self) -> int:
        if self.records is None:
            return 0
        return len(self.records)

class AuxiliaryDataColumns:
    def __init__(
            self,
            columns: list[AuxiliaryDataColumn]):
        self.columns = columns

    def _dump_column_info(self) -> list[dict]:
        return [
            {
                "column_name": column.column_name,
                "is_date": column.is_date,
                "is_label": column.is_label,
                "is_train_data": column.is_train_data,
                "is_generated": column.is_generated,
            }
            for column in self.columns
        ]

    def _dump_records(self) -> np.ndarray:
        """AuxiliaryDataColumns の records を numpy.ndarray として返す
        
        Returns:
            np.ndarray: [description]
            
        Notes:
            - 各カラムの records は同じ長さであることを前提としている
            - 各カラムの records は list であることを前提としている
        """
        return np.array([column.records for column in self.columns])
    
    def dump(self) -> dict:
        return {
            "info": self._dump_column_info(),
            "data": self._dump_records(),
        }

    @staticmethod
    def load(dumped_obj: dict) -> AuxiliaryDataColumns:
        aux_info: list[dict[str, str|int|bool]] = dumped_obj["info"]
        aux_data: np.ndarray = dumped_obj["data"]
        columns = []
        for icolumn, column_info in enumerate(aux_info):
            column = AuxiliaryDataColumn(**column_info)
            column.records = aux_data[icolumn]
            columns.append(column)
        return AuxiliaryDataColumns(columns)

    @property
    def columns(self) -> list[AuxiliaryDataColumn]:
        return self._columns

    @columns.setter
    def columns(self, columns_given: list[AuxiliaryDataColumn]):
        # columns の 各要素のcolumn_nameがユニークとなるようにしたうえで保持する
        column_names_original = [column.column_name for column in columns_given]
        unique_column_names = self.get_unique_column_names(column_names_original)
        for column_name_new, column in zip(unique_column_names, columns_given):
            column.column_name = column_name_new
        self._columns = columns_given

    def get_unique_column_names(self, column_names_given: list[str]) -> list[str]:
        # 重複しているカラムのindexとカラム名を保持する
        column_names_duplicated = [
            (icolumn, column_name)
            for icolumn, column_name in enumerate(column_names_given)
            if column_names_given.count(column_name) > 1
        ]

        # 各indexに対して、新しいカラム名の辞書を作成する。
        # 重複していないカラムはそのままの名前を保持するようにする
        d_new_column_name = {
            icolumn: column_name
            for icolumn, column_name in enumerate(column_names_given)
            if column_names_given.count(column_name) == 1
        }
        # 重複しているカラムは、新しいカラム名を作成する
        for icolumn, column_name in column_names_duplicated:
            is_duplicated = True
            while is_duplicated:
                new_name_test = f"{column_name}_{icolumn:02d}"
                if new_name_test not in d_new_column_name.values():
                    d_new_column_name[icolumn] = new_name_test
                    is_duplicated = False
        # d_new_column_name は int -> str の辞書なので、index順に並べ替えて str -> str の辞書にして返す
        return [
            d_new_column_name[icolumn] for icolumn in range(len(column_names_given))
        ]
    
    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, exclude_columns: list[str] = None):
        columns = []
        column_names = [column_name for column_name in df.columns if column_name not in exclude_columns]
        for column_name in column_names:
            column = AuxiliaryDataColumn(column_name, df[column_name].tolist())
            columns.append(column)
        return cls(columns)

    @property
    def date_column_names(self) -> list[str]:
        return [column.column_name for column in self.columns if column.is_date]

    @property
    def label_column_names(self) -> list[str]:
        return [column.column_name for column in self.columns if column.is_label]

    @property
    def train_data_column_names(self) -> list[str]:
        return [column.column_name for column in self.columns if column.is_train_data]

    @property
    def generated_column_names(self) -> list[str]:
        return [column.column_name for column in self.columns if column.is_generated]

    @property
    def column_names(self) -> list[str]:
        return [column.column_name for column in self.columns]
    
    def get_column(self, column_name: str) -> AuxiliaryDataColumn:
        return [column for column in self.columns if column.column_name == column_name][0]
    
    def __getitem__(self, column_name: str) -> AuxiliaryDataColumn:
        return self.get_column(column_name)
    
    def __len__(self) -> int:
        return len(self.columns[0])

    def shape(self) -> tuple[int, int]:
        return (len(self.columns), len(self))

    def __iter__(self)->Iterator[AuxiliaryDataColumn]:
        return iter(self.columns)
    
    def __repr__(self) -> str:
        out = f"AuxiliaryDataColumns(\n"
        for column in self.columns:
            out += f"    {column},\n"
        out += ")"
        return out

from __future__ import annotations
from typing import Callable, overload
import streamlit_antd_components as sac
class Tree:
    DELIMITER = "/"
    LEAF_MARKER = "::LEAF::"
    def __init__(
            self,
            label,
            index: int = -1,
            children: None|list[Tree] = None,
            description: None|str = None,
            checked: bool = False,
            disabled: bool = False,
            tag: None|str = None,
            ):
        self.label = label
        self.index = index
        self.checked = checked
        self.disabled = disabled
        self.description = description
        self.tag = tag
        if children is None:
            self.children = None
        else:
            self.children = [
                Tree(
                    label=f"{self.label}{Tree.DELIMITER}{child.label}",
                    index=self.index + 1 + ichild,
                    children=child.children,
                    checked=self.checked or child.checked,
                    disabled=self.disabled or child.disabled,
                    tag=child.tag,
                    description=child.description,
                    ) for ichild, child in enumerate(children)]

    def get_checked_index(self, tree_list:list[Tree]=None)->list[int]:
        checked_index = []
        for itree, tree in enumerate(tree_list):
            if tree.checked:
                checked_index.append(itree)
        return checked_index

    def get_open_index(self, tree_list:list[Tree]=None)->list[int]:
        open_index = []
        for itree, tree in enumerate(tree_list):
            if tree.recursively_find_checked_index():
                open_index.append(itree)
        return open_index
    
    def recursively_find_checked_index(self):
        if self.children is None:
            return self.checked
        else:
            return self.checked or any([child.recursively_find_checked_index() for child in self.children])

    def get_flat_tree_list(self)->list[Tree]:
        return self.recursively_get_flat_tree_list()[1:]

    def recursively_get_flat_tree_list(self)->list[Tree]:
        if self.children is None:
            return [self]
        else:
            return [self] + [child for child in self.children for child in child.recursively_get_flat_tree_list()]
        
    @property
    def icon(self):
        if self.children:
            return "folder"
        else:
            return "table"

    @overload
    def convert_recursively(self, func: Callable[[Tree], sac.TreeItem], _called_by_root:bool=False)->list[sac.TreeItem]:
        ...

    @overload
    def convert_recursively(self, func: Callable[[Tree], sac.CasItem], _called_by_root:bool=False)->list[sac.CasItem]:
        ...

    def convert_recursively(self, func, _called_by_root:bool=False):
        if _called_by_root:
            self._checked_index = []
        if self.children is None:
            tree_items = []
        else:
            tree_items = [func(child) for child in self.children]
        return tree_items
    
    def convert_to_sac_tree_item(self)->sac.TreeItem:
        return sac.TreeItem(
            label = self.label if self.children else f"{self.label}{Tree.LEAF_MARKER}",
            icon = self.icon,
            children = self.convert_recursively(Tree.convert_to_sac_tree_item, _called_by_root=False) if self.children else None,
            disabled=self.disabled,
            tooltip=self.description,
            tag=self.tag,
        )
    
    def convert_to_sac_cas_item(self)->sac.CasItem:
        return sac.CasItem(
            label = self.label if self.children else f"{self.label}{Tree.LEAF_MARKER}",
            icon = self.icon,
            children = self.convert_recursively(Tree.convert_to_sac_cas_item, _called_by_root=False) if self.children else None,
            disabled=self.disabled,
        )
    
    def to_sac_tree_items(self)->tuple[list[sac.TreeItem], list[int], list[int]]:
        items = self.convert_recursively(Tree.convert_to_sac_tree_item, _called_by_root=True)
        tree_list = self.get_flat_tree_list()
        checked_index = self.get_checked_index(tree_list)
        open_index = self.get_open_index(tree_list)
        return items, checked_index, open_index
    
    def to_sac_cascade_items(self)->tuple[list[sac.CasItem], list[int], list[int]]:
        items = self.convert_recursively(Tree.convert_to_sac_cas_item, _called_by_root=True)
        checked_index = self.get_checked_index(tree_list)
        open_index = self.get_open_index(tree_list)
        return items, checked_index, open_index
    
    @classmethod
    def format_func(cls, s:str)->str:
        return s.split("/")[-1].replace(cls.LEAF_MARKER, "")
    
    @classmethod
    def filter_to_leaf(cls, selected_nodes:list[str])->list[str]:
        return [cls.format_func(node) for node in selected_nodes if cls.LEAF_MARKER in node]

DEMO_TREE = Tree(
    label = "root",
    children = [
        Tree(
            label = "child1",
            children = [
                Tree(
                    label = "child1-1",
                    checked=True,
                    disabled=True,
                    description="デフォルトで付与されます",
                    tag="default",
                    children = [
                        Tree(
                            label = "child1-1-1",

                        ),
                        Tree(
                            label = "child1-1-2",
                        ),
                    ]
                ),
                Tree(
                    label = "child1-2",
                ),
            ]
        ),
        Tree(
            label = "child2",
        ),
        Tree(
            label = "child3",
            children = [
                Tree(
                    label = "child3-1",
                    checked=True,
                ),
                Tree(
                    label = "child3-2",
                ),
                Tree(
                    label = "child3-3",
                ),
            ]
        ),
        Tree(
            label = "child4",
            children = [
                Tree(
                    label = "child4-1",
                    # checked=True,
                    children = [
                        Tree(
                            label = "child4-1-1",
                            checked=True,
                            disabled=True,
                            description="デフォルトで付与されます",
                            tag="default",
                        ),
                        Tree(
                            label = "child4-1-2",
                        ),
                    ]
                ),
                Tree(
                    label = "child4-2",
                ),
                Tree(
                    label = "child4-3",
                ),
            ]
        ),
        Tree(
            label = "child5",
        ),
        Tree(
            label = "child6",
        ),
        Tree(
            label = "child7",
        ),
    ],
)
import streamlit as st
# print("*****************")
sac_tree_items, checked_index, open_index = DEMO_TREE.to_sac_tree_items()

# st.write(checked_index)
# st.write(sac_tree_items)

selected_tree = sac.tree(
    label = "demo_tree",
    items = sac_tree_items,
    checkbox = True,
    open_index=open_index,
    index=checked_index,
    format_func=Tree.format_func,
)
selected_leaf = Tree.filter_to_leaf(selected_tree)
st.write(selected_leaf)
# st.write(Tree.filter_to_leaf(selected_tree))


# st.write([f"{tree.checked}, {tree.label}" for tree in tree_list])
# st.write(checked_index)
# sac_cascade_items, checked_index = DEMO_TREE.to_sac_cascade_items()

# selected_cascade = sac.cascader(
#     label = "demo_cascade",
#     items = sac_cascade_items,
#     format_func=Tree.format_func,
#     multiple=True,
#     search=True,
#     clear=True,
# )

# st.write(selected_cascade)
# selected_leaf = Tree.filter_to_leaf(selected_cascade)
# st.write(selected_leaf)


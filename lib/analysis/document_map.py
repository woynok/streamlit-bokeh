import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from umap import UMAP

class DocumentMap:
    """2次元上の点として文書を表現するための位置やラベルを持つクラス

    Args:
        embeddings (np.ndarray): 文書の埋め込みベクトル
        max_cluster (int, optional): クラスタ数の上限. 2以上の整数である必要あり。Defaults to 10.
        embeddings_norm (np.ndarray): 文書の埋め込みベクトルを正規化したもの. 距離をとると cos similarity に proportion する

    Attributes:
        max_cluster (int): クラスタ数の上限
        filterd_indices (dict): Outlier 除去などをした場合にこのMapで使っている文書のインデックスを保存する
        xs (np.ndarray): 2次元に削減した埋め込みベクトルのx座標
        ys (np.ndarray): 2次元に削減した埋め込みベクトルのy座標
        dict_labels (dict[int, np.ndarray]): n_clustersをkeyに、そのクラスタに属する文書のインデックスの辞書
    """

    def __init__(self, embeddings: np.ndarray, max_cluster: int = 14):
        if max_cluster < 2 or not isinstance(max_cluster, int):
            raise ValueError(
                "max_cluster must be an integer greater than or equal to 2"
            )
        self.embeddings = embeddings
        # cosine similarityで距離を計算するために正規化する
        self.embeddings_norm = self.embeddings / np.linalg.norm(self.embeddings, axis = 1, keepdims = True)
        self.max_cluster = max_cluster
        self.filterd_indices = None  # Outlier 除去などをした場合にこのMapで使っている文書のインデックスを保存する
        self.dict_labels: dict[int, np.ndarray] = None
        self.xs, self.ys = self._reduce_dimension()
        self.dict_labels: dict[int, np.ndarray] = self._clustering_all()
    
    def _reduce_dimension(self):
        """文書の埋め込みベクトルを2次元に削減する

        Args:
            embeddings (np.ndarray): 文書の埋め込みベクトル

        Returns:
            np.ndarray: 2次元に削減した埋め込みベクトル
        """
        reducer = UMAP(n_components=2, min_dist = 0.3, metric="cosine", random_state=0, n_jobs = 1)
        embeddings_reduced = reducer.fit_transform(self.embeddings)
        return embeddings_reduced[:, 0], embeddings_reduced[:, 1]

    def _clustering_all(self)->dict[int, np.ndarray]:
        """文書の埋め込みベクトルをクラスタリングする

        Args:
            embeddings (np.ndarray): 文書の埋め込みベクトル

        Returns:
            dict: n_clustersをkeyに、そのクラスタに属する文書のインデックスの辞書
        """
        dict_labels = {}
        for n_clusters in range(2, self.max_cluster + 1):
            kmeans_model = KMeans(
                n_clusters=n_clusters,
                random_state=0,
                n_init="auto",
                init="k-means++",
            ).fit(self.embeddings_norm)
            dict_labels[n_clusters] = kmeans_model.labels_
        return dict_labels

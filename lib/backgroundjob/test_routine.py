import numpy as np
from umap import UMAP
from datetime import datetime
def test_routine():
    print(f'{datetime.now()} Test routine start')
    x_in = np.random.rand(2000, 800)
    reducer = UMAP(
        n_neighbors=5,
        min_dist=0.3,
        metric='euclidean',
        n_components=2,
        random_state=0,
        n_jobs=1,
    )
    reducer.fit(x_in)
    x_out = reducer.transform(x_in)
    print("writing result.txt...")
    with open("result.txt", "w") as f:
        f.write("Embedding shape: " + str(x_out.shape) + "\n")
        f.write(f"{datetime.now()} Test routine end\n")
    print("finished")
    
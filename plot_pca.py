# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "fastcan",
#     "matplotlib",
#     "nonlinear-benchmarks",
# ]
#
# [tool.uv.sources]
# fastcan = { git = "https://github.com/scikit-learn-contrib/fastcan" }
# ///

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
import nonlinear_benchmarks
import matplotlib.pyplot as plt

from utils import get_narx_terms


def _plot_pca(u, y, figure_name):
    poly_terms, _ = get_narx_terms(u, y)
    pca = PCA(2).fit(poly_terms)
    pcs_all = pca.transform(poly_terms)

    kmeans = MiniBatchKMeans(
        n_clusters=100,
        random_state=0,
        batch_size=6,
        n_init="auto",
    ).fit(poly_terms)
    atoms = kmeans.cluster_centers_
    pcs_fastcan = pca.transform(atoms)

    rng = np.random.default_rng(123)
    ids_random = rng.choice(y.size, 100, replace=False)
    pcs_random = pca.transform(poly_terms[ids_random])

    plt.scatter(pcs_all[:, 0], pcs_all[:, 1], s=5)
    plt.scatter(pcs_fastcan[:, 0], pcs_fastcan[:, 1], s=50, marker="o", alpha=0.9)
    plt.scatter(pcs_random[:, 0], pcs_random[:, 1], s=30, marker="*", alpha=0.9)
    plt.xlabel("The First Principle Component")
    plt.ylabel("The Second Principle Component")
    plt.legend(["All data", "FastCan pruned", "Random pruned"])
    plt.savefig(figure_name, bbox_inches="tight")
    plt.close()


def main() -> None:
    train_val, _ = nonlinear_benchmarks.EMPS()
    train_val_u, train_val_y = train_val
    _plot_pca(train_val_u, train_val_y, "pca_emps.png")

    train_val, _ = nonlinear_benchmarks.WienerHammerBenchMark()
    train_val_u, train_val_y = train_val
    _plot_pca(train_val_u, train_val_y, "pca_whbm.png")


if __name__ == "__main__":
    main()
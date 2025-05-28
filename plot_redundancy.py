"""Visualize the redundancy of the dataset."""

import matplotlib.pyplot as plt
import numpy as np
from fastcan.narx import make_time_shift_features, make_time_shift_ids
from sklearn.decomposition import PCA

if __name__ == "__main__":
    # Plot raw time-sereis data
    u = np.linspace(0, 1, 100)
    y = np.sin(2 * np.pi * u)

    plt.plot(u, y)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.xlim(0, 1)
    plt.savefig("redundancy_raw_data.png", bbox_inches="tight")
    plt.close()

    # Plot the redundancy in column-wise (feature-wise)
    time_ids = make_time_shift_ids(1, 20, True)
    Y = make_time_shift_features(y.reshape(-1, 1), time_ids)
    Y_corr = np.corrcoef(Y, rowvar=False)

    plt.imshow(Y_corr)
    cb = plt.colorbar()
    cb.ax.set_title("$r$")
    ticks = np.arange(0, len(time_ids), step=5)
    plt.xticks(ticks)
    plt.yticks(ticks)
    plt.xlabel("Feature Index")
    plt.ylabel("Feature Index")
    plt.savefig("redundancy_feature.png", bbox_inches="tight")
    plt.close()

    # Plot the redundancy in row-wise (sample-wise)
    pca = PCA(2).fit(Y)

    pcs = pca.transform(Y)

    plt.scatter(pcs[:, 0], pcs[:, 1], c=u)
    cb = plt.colorbar()
    cb.ax.set_title("Time (s)")
    plt.xlabel("The First Principle Component")
    plt.ylabel("The Second Principle Component")
    plt.savefig("redundancy_sample.png", bbox_inches="tight")
    plt.close()

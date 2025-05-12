"""Compare the performance of different batch sizes"""

import click
import matplotlib.pyplot as plt
import nonlinear_benchmarks
import numpy as np
from sklearn.metrics import r2_score

from utils import fastcan_pruned_narx, get_dual_stable_equilibria_data, get_narx_terms


def _plot_batch(u, y, batch_size_list, n_samples, n_atoms, figure_name):
    poly_terms, y, narx = get_narx_terms(u, y)

    n_random = 10
    n_batches = len(batch_size_list)
    r2_fastcan = np.zeros((n_random, n_batches))
    for i in range(n_random):
        print(figure_name, "   ", f"Random test: {i+1}/{n_random}")
        for j, batch_size in enumerate(batch_size_list):
            coef, intercept = fastcan_pruned_narx(
                poly_terms, y, n_samples, i, batch_size, n_atoms
            )
            r2_fastcan[i, j] = r2_score(
                np.r_[coef, intercept], np.r_[narx.coef_, narx.intercept_]
            )

    plt.boxplot(r2_fastcan, tick_labels=batch_size_list)
    plt.ylabel("R2")
    plt.xlabel("Batch Size")

    plt.savefig(figure_name, bbox_inches="tight")
    plt.close()
    print("Image " + figure_name + " has been generated.")


@click.command()
@click.option("--dataset", default="dsed", help="Choose dataset from: dsed, emps, whbm")
def main(dataset) -> None:
    match dataset:
        case "dsed":
            train_val_u, train_val_y, _ = get_dual_stable_equilibria_data()
            _plot_batch(
                train_val_u,
                train_val_y,
                # np.linspace(2, 6, 5, dtype=int),
                np.linspace(2, 10, 5, dtype=int),
                600,
                60,
                "batch_dsed.png",
            )
        case "emps":
            train_val, _ = nonlinear_benchmarks.EMPS()
            train_val_u, train_val_y = train_val
            _plot_batch(
                train_val_u,
                train_val_y,
                # np.linspace(10, 100, 10, dtype=int),
                np.linspace(1, 10, 10, dtype=int),
                10000,
                700,
                "batch_emps.png",
            )
        case "whbm":
            train_val, _ = nonlinear_benchmarks.WienerHammerBenchMark()
            train_val_u, train_val_y = train_val
            _plot_batch(
                train_val_u,
                train_val_y,
                np.linspace(10, 100, 10, dtype=int),
                10000,
                100,
                "batch_whbm.png",
            )
        case _:
            raise NameError(
                "The dataset is not supported. Please choose from: dsed, emps, whbm"
            )


if __name__ == "__main__":
    main()

"""Compare the performance of different number of atoms"""

import click
import matplotlib.pyplot as plt
import nonlinear_benchmarks
import numpy as np
from sklearn.metrics import r2_score

from utils import fastcan_pruned_narx, get_dual_stable_equilibria_data, get_narx_terms


def _plot_atom(u, y, n_atoms_list, n_samples, figure_name):
    poly_terms, y, narx = get_narx_terms(u, y)

    n_random = 10
    r2_fastcan = np.zeros((n_random, len(n_atoms_list)))
    for i in range(n_random):
        print(figure_name, "   ", f"Random test: {i+1}/{n_random}")
        for j, n_atoms in enumerate(n_atoms_list):
            coef, intercept = fastcan_pruned_narx(
                poly_terms,
                y,
                n_samples,
                i,
                n_atoms=n_atoms,
            )
            r2_fastcan[i, j] = r2_score(
                np.r_[coef, intercept], np.r_[narx.coef_, narx.intercept_]
            )

    plt.boxplot(r2_fastcan, tick_labels=n_atoms_list)
    plt.ylabel("R2")
    plt.xlabel("No. of Atoms")

    plt.savefig(figure_name, bbox_inches="tight")
    plt.close()
    print("Image " + figure_name + " has been generated.")


@click.command()
@click.option("--dataset", default="dsed", help="Choose dataset from: dsed, emps, whbm")
def main(dataset) -> None:
    match dataset:
        case "dsed":
            train_val_u, train_val_y, _ = get_dual_stable_equilibria_data()
            _plot_atom(
                train_val_u,
                train_val_y,
                [5, 10, 30, 60, 100, 200],
                600,
                "atom_dsed.png",
            )
        case "emps":
            train_val, _ = nonlinear_benchmarks.EMPS()
            train_val_u, train_val_y = train_val
            _plot_atom(
                train_val_u,
                train_val_y,
                [10, 40, 70, 100, 400, 700, 1000, 2000],
                10000,
                "atom_emps.png",
            )
        case "whbm":
            train_val, _ = nonlinear_benchmarks.WienerHammerBenchMark()
            train_val_u, train_val_y = train_val
            _plot_atom(
                train_val_u,
                train_val_y,
                # [10, 100, 1000],
                [10, 40, 70, 100, 400, 700, 1000],
                10000,
                "atom_whbm.png",
            )
        case _:
            raise NameError(
                "The dataset is not supported. Please choose from: dsed, emps, whbm"
            )


if __name__ == "__main__":
    main()

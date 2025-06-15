"""Generate box plot"""

import click
import matplotlib.pyplot as plt
import nonlinear_benchmarks
import numpy as np
from rich.progress import track

from utils import (
    fastcan_pruned_narx,
    get_dsed_eq,
    get_dsed_tr,
    get_dual_stable_equilibria_data,
    get_narx_terms,
    get_r2,
    random_pruned_narx,
)


def _plot_box(
    u,
    y,
    n_samples_to_select,
    n_atoms,
    twinx,
    figure_name,
    n_random,
    intercept=True,
    max_delay=10,
):
    poly_terms, y, narx = get_narx_terms(u, y, intercept, max_delay)

    r2_fastcan = np.zeros(n_random)
    r2_random = np.zeros(n_random)
    for i in track(range(n_random), description="[green]Random test..."):
        coef = fastcan_pruned_narx(
            poly_terms, y, n_samples_to_select, i, n_atoms=n_atoms, intercept=intercept
        )
        r2_fastcan[i] = get_r2(coef, narx)

        coef = random_pruned_narx(
            poly_terms, y, n_samples_to_select, i, intercept=intercept
        )
        r2_random[i] = get_r2(coef, narx)
    fig, ax1 = plt.subplots()
    ax1.boxplot(r2_fastcan.reshape(-1, 1), positions=[1])
    if twinx:
        ax2 = ax1.twinx()
        ax2.boxplot(r2_random.reshape(-1, 1), positions=[2])
        ax2.set_ylabel("R2")
    else:
        ax1.boxplot(r2_random.reshape(-1, 1), positions=[2])
    ax1.set_ylabel("R2")
    ax1.set_xticks(ticks=[1, 2], labels=["FastCan", "Random"])

    fig.savefig(figure_name, bbox_inches="tight")
    plt.close()
    print("Image " + figure_name + " has been generated.")


@click.command()
@click.option(
    "--dataset",
    default="dsed",
    help="Choose dataset from: dsed, emps, whbm, dsed-eq, dsed-tr",
)
@click.option("--n_random", default=10, help="Set the number of random tests")
def main(dataset, n_random) -> None:
    match dataset:
        case "dsed-eq":
            u, y = get_dsed_eq()
            _plot_box(u, y, 100, 20, False, "box_dsed_eq.png", n_random=10, max_delay=4)
        case "dsed-tr":
            u, y = get_dsed_tr()
            _plot_box(u, y, 100, 20, False, "box_dsed_tr.png", n_random=10, max_delay=6)
        case "dsed":
            train_val_u, train_val_y, _ = get_dual_stable_equilibria_data()
            _plot_box(
                train_val_u,
                train_val_y,
                100,
                15,
                False,
                "box_dsed.png",
                n_random=n_random,
                max_delay=4,
            )
        case "emps":
            train_val, _ = nonlinear_benchmarks.EMPS()
            train_val_u, train_val_y = train_val
            # No intercept for EMPS dataset
            _plot_box(
                train_val_u,
                train_val_y,
                100,
                10,
                False,
                "box_emps.png",
                n_random=n_random,
                max_delay=4,
            )
        case "whbm":
            train_val, _ = nonlinear_benchmarks.WienerHammerBenchMark()
            train_val_u, train_val_y = train_val
            _plot_box(
                train_val_u,
                train_val_y,
                100,
                5,
                False,
                "box_whbm.png",
                n_random=n_random,
                max_delay=7,
            )
        case _:
            raise NameError(
                "The dataset is not supported. Please choose from: dsed, emps, whbm"
            )


if __name__ == "__main__":
    main()

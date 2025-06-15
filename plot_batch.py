"""Compare the performance of different batch sizes"""

import click
import matplotlib.pyplot as plt
import nonlinear_benchmarks
import numpy as np
from rich.progress import Progress, TimeRemainingColumn

from utils import (
    fastcan_pruned_narx,
    get_dsed_eq,
    get_dsed_tr,
    get_dual_stable_equilibria_data,
    get_narx_terms,
    get_r2,
)


def _plot_batch(
    u,
    y,
    batch_size_list,
    n_samples,
    n_atoms,
    n_random,
    figure_name,
    intercept=True,
    max_delay=10,
):
    poly_terms, y, narx = get_narx_terms(u, y, intercept, max_delay)

    """Plot the R2 for different batch sizes."""
    n_batches = len(batch_size_list)
    r2_fastcan = np.zeros((n_random, n_batches))
    columns = [*Progress.get_default_columns()]
    columns[-1] = TimeRemainingColumn(elapsed_when_finished=True)
    with Progress(*columns, auto_refresh=True) as pb:
        t1 = pb.add_task("[red]Pruning data...", total=n_batches)
        t2 = pb.add_task("[green]Random test...", total=n_random)
        for i in range(n_random):
            for j, batch_size in enumerate(batch_size_list):
                coef = fastcan_pruned_narx(
                    poly_terms, y, n_samples, i, batch_size, n_atoms, intercept
                )
                r2_fastcan[i, j] = get_r2(coef, narx)
                pb.update(task_id=t1, completed=j + 1)
            pb.update(task_id=t2, completed=i + 1)

    plt.boxplot(r2_fastcan, tick_labels=batch_size_list)
    plt.ylabel("R2")
    plt.xlabel("Batch Size")

    plt.savefig(figure_name, bbox_inches="tight")
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
            _plot_batch(
                u,
                y,
                np.linspace(1, 5, 5, dtype=int),
                100,
                20,
                n_random=n_random,
                figure_name="batch_dsed_eq.png",
                max_delay=4,
            )
        case "dsed-tr":
            u, y = get_dsed_tr()
            _plot_batch(
                u,
                y,
                np.linspace(1, 5, 5, dtype=int),
                100,
                20,
                n_random=n_random,
                figure_name="batch_dsed_tr.png",
                max_delay=6,
            )
        case "dsed":
            train_val_u, train_val_y, _ = get_dual_stable_equilibria_data()
            _plot_batch(
                train_val_u,
                train_val_y,
                # np.linspace(2, 6, 5, dtype=int),
                np.linspace(1, 7, 7, dtype=int),
                100,
                15,
                n_random=n_random,
                figure_name="batch_dsed.png",
                max_delay=4,
            )
        case "emps":
            train_val, _ = nonlinear_benchmarks.EMPS()
            train_val_u, train_val_y = train_val
            _plot_batch(
                train_val_u,
                train_val_y,
                np.linspace(1, 10, 10, dtype=int),
                100,
                10,
                # np.linspace(5, 50, 10, dtype=int),
                # 100,
                # 2,
                # np.linspace(1, 20, 10, dtype=int),
                # 100,
                # 5,
                # np.linspace(1, 5, 5, dtype=int),
                # 100,
                # 20,
                n_random=n_random,
                figure_name="batch_emps.png",
                # intercept=False,  # No intercept for EMPS dataset
                max_delay=4,
            )
        case "whbm":
            train_val, _ = nonlinear_benchmarks.WienerHammerBenchMark()
            train_val_u, train_val_y = train_val
            _plot_batch(
                train_val_u,
                train_val_y,
                np.linspace(2, 20, 10, dtype=int),
                100,
                5,
                n_random=n_random,
                figure_name="batch_whbm.png",
                max_delay=7,
            )
        case _:
            raise NameError(
                "The dataset is not supported. Please choose from: dsed, emps, whbm"
            )


if __name__ == "__main__":
    main()

"""Compare the performance of different number of atoms"""

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


def _plot_atom(
    u, y, n_atoms_list, n_samples, n_random, figure_name, intercept=True, max_delay=10
):
    poly_terms, y, narx = get_narx_terms(u, y, intercept, max_delay)

    """Plot the R2 for different number of atoms."""
    n_tests = len(n_atoms_list)
    r2_fastcan = np.zeros((n_random, n_tests))
    columns = [*Progress.get_default_columns()]
    columns[-1] = TimeRemainingColumn(elapsed_when_finished=True)
    with Progress(*columns, auto_refresh=True) as pb:
        t1 = pb.add_task("[red]Pruning data...", total=n_tests)
        t2 = pb.add_task("[green]Random test...", total=n_random)
        for i in range(n_random):
            for j, n_atoms in enumerate(n_atoms_list):
                coef = fastcan_pruned_narx(
                    poly_terms,
                    y,
                    n_samples,
                    i,
                    n_atoms=n_atoms,
                    intercept=intercept,
                )
                r2_fastcan[i, j] = get_r2(coef, narx)
                pb.update(task_id=t1, completed=j + 1)
            pb.update(task_id=t2, completed=i + 1)

    plt.boxplot(r2_fastcan, tick_labels=n_atoms_list)
    plt.ylabel("R2")
    plt.xlabel("No. of Atoms")

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
            _plot_atom(
                u,
                y,
                # [3, 5, 10, 15, 20, 25, 30, 50],
                [2, 5, 10, 15, 20, 30, 50, 70],
                100,
                n_random=n_random,
                figure_name= "atom_dsed_eq.png",
                intercept=True,
                max_delay=4,
            )
        case "dsed-tr":
            u, y = get_dsed_tr()
            _plot_atom(
                u,
                y,
                [2, 5, 10, 15, 20, 30, 50, 70],
                100,
                n_random=n_random,
                figure_name="atom_dsed_tr.png",
                intercept=True,
                max_delay=6,
            )
        case "dsed":
            train_val_u, train_val_y, _ = get_dual_stable_equilibria_data()
            _plot_atom(
                train_val_u,
                train_val_y,
                [2, 5, 10, 15, 20, 30, 50, 70],
                100,
                n_random=n_random,
                figure_name="atom_dsed.png",
                max_delay=4,
            )
        case "emps":
            train_val, _ = nonlinear_benchmarks.EMPS()
            train_val_u, train_val_y = train_val
            _plot_atom(
                train_val_u,
                train_val_y,
                [2, 5, 10, 15, 20, 30, 50, 70],
                100,
                n_random=n_random,
                figure_name="atom_emps.png",
                # intercept=False,  # No intercept for EMPS dataset
                max_delay=4,
            )
        case "whbm":
            train_val, _ = nonlinear_benchmarks.WienerHammerBenchMark()
            train_val_u, train_val_y = train_val
            _plot_atom(
                train_val_u,
                train_val_y,
                [2, 5, 10, 15, 20, 30, 50, 70],
                100,
                n_random=n_random,
                figure_name="atom_whbm.png",
                max_delay=7,
            )
        case _:
            raise NameError(
                "The dataset is not supported. Please choose from: dsed, emps, whbm"
            )


if __name__ == "__main__":
    main()

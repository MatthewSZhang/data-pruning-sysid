"""Generate atom vs. sample plot"""

import click
import matplotlib.pyplot as plt
import nonlinear_benchmarks
import numpy as np
from matplotlib.colors import LogNorm
from rich.progress import Progress, TimeRemainingColumn

from utils import (
    fastcan_pruned_narx,
    get_dual_stable_equilibria_data,
    get_narx_terms,
    get_r2,
)


def _plot_sample(
    u,
    y,
    n_sample_lower,
    n_sample_upper,
    n_sample_steps,
    # n_atom_lower,
    # n_atom_upper,
    # n_atom_steps,
    atom_step,
    figure_name,
    intercept=True,
    n_random=5,
):
    poly_terms, y, narx = get_narx_terms(u, y, intercept)
    sample_step = int((n_sample_upper - n_sample_lower) / (n_sample_steps - 1))
    # atom_step = int((n_atom_upper - n_atom_lower) / (n_atom_steps - 1))
    n_atom_steps = len(atom_step)
    r2_fastcan = np.zeros((n_random, n_sample_steps, n_atom_steps))
    columns = [*Progress.get_default_columns()]
    columns[-1] = TimeRemainingColumn(elapsed_when_finished=True)
    with Progress(*columns, auto_refresh=True) as pb:
        t1 = pb.add_task("[yellow]Atom test...", total=n_atom_steps)
        t2 = pb.add_task("[red]Sample test...", total=n_sample_steps)
        t3 = pb.add_task("[green]Random test...", total=n_random)
        for i in range(n_random):
            for j in range(n_sample_steps):
                for k, n_atoms in enumerate(atom_step):
                    coef = fastcan_pruned_narx(
                        poly_terms,
                        y,
                        n_samples_to_select=j * sample_step + n_sample_lower,
                        random_state=i,
                        # n_atoms=k * atom_step + n_atom_lower,
                        n_atoms=n_atoms,
                        intercept=intercept,
                    )
                    r2_fastcan[i, j, k] = get_r2(coef, narx)
                    pb.update(task_id=t1, completed=k + 1)
                pb.update(task_id=t2, completed=j + 1)
            pb.update(task_id=t3, completed=i + 1)

    np.save(figure_name[:-4], r2_fastcan)
    print("Results have been saved to " + figure_name[:-4] + ".npy")
    # Generate plot
    r2_mean = 1 - r2_fastcan.mean(axis=0)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(
        r2_mean.T,
        cmap="jet",
        aspect="auto",
        origin="lower",
        norm=LogNorm(vmin=r2_mean.min(), vmax=r2_mean.max()),
    )
    plt.colorbar(im, ax=ax, label="1-R2")

    sample_ticks = np.linspace(
        n_sample_lower, n_sample_upper, n_sample_steps, endpoint=True
    )
    # atom_ticks = np.linspace(n_atom_lower, n_atom_upper, n_atom_steps, endpoint=True)
    atom_ticks = atom_step

    ax.set_xticks(range(n_sample_steps))
    ax.set_xticklabels([f"{int(x)}" for x in sample_ticks])
    ax.set_yticks(range(n_atom_steps))
    ax.set_yticklabels([f"{int(x)}" for x in atom_ticks])
    ax.set_xlabel("Number of Samples")
    ax.set_ylabel("Number of Atoms")

    plt.tight_layout()
    fig.savefig(figure_name, bbox_inches="tight")
    plt.close()
    print("Image " + figure_name + " has been generated.")


@click.command()
@click.option("--dataset", default="dsed", help="Choose dataset from: dsed, emps, whbm")
@click.option("--n_random", default=1, help="Set the number of random tests")
def main(dataset, n_random) -> None:
    match dataset:
        case "dsed":
            train_val_u, train_val_y, _ = get_dual_stable_equilibria_data()
            _plot_sample(
                train_val_u,
                train_val_y,
                n_sample_lower=300,
                n_sample_upper=600,
                n_sample_steps=11,
                # n_atom_lower=3,
                # n_atom_upper=120,
                # n_atom_steps=40,
                atom_step = [5, 10, 30, 60, 100, 150, 200],
                figure_name="sample_dsed.png",
                intercept=True,
                n_random=n_random,
            )
        case "emps":
            train_val, _ = nonlinear_benchmarks.EMPS()
            train_val_u, train_val_y = train_val
            # No intercept for EMPS dataset
            _plot_sample(
                train_val_u,
                train_val_y,
                n_sample_lower=2000,
                n_sample_upper=20000,
                n_sample_steps=10,
                # n_atom_lower=200,
                # n_atom_upper=1200,
                # n_atom_steps=6,
                atom_step=[10, 40, 70, 100, 400, 700, 1000, 2000],
                figure_name="sample_emps.png",
                intercept=False,
                n_random=n_random,
            )
        case "whbm":
            train_val, _ = nonlinear_benchmarks.WienerHammerBenchMark()
            train_val_u, train_val_y = train_val
            _plot_sample(
                train_val_u,
                train_val_y,
                n_sample_lower=2000,
                n_sample_upper=20000,
                n_sample_steps=10,
                # n_atom_lower=200,
                # n_atom_upper=1200,
                # n_atom_steps=6,
                atom_step=[10, 40, 70, 100, 400, 700, 1000, 2000],
                figure_name="sample_whbm.png",
                intercept=True,
                n_random=n_random,
            )
        case _:
            raise NameError(
                "The dataset is not supported. Please choose from: dsed, emps, whbm"
            )


if __name__ == "__main__":
    main()

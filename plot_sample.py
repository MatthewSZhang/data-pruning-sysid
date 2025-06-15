"""Generate atom vs. sample plot"""

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
    n_random,
    intercept=True,
    max_delay=10,
):
    poly_terms, y, narx = get_narx_terms(u, y, intercept, max_delay)
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
    # r2_mean = 1 - r2_fastcan.mean(axis=0)
    r2_mean = r2_fastcan.mean(axis=0)
    fonts = 15
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(
        r2_mean.T,
        cmap="jet",
        aspect="auto",
        origin="lower",
        # norm=LogNorm(vmin=r2_mean.min(), vmax=r2_mean.max()),
    )
    cbar = plt.colorbar(im, ax=ax)
    # cbar.set_label("1-R2", fontsize=12)
    cbar.set_label("R2", fontsize=12)
    cbar.ax.tick_params(labelsize=12)

    sample_ticks = np.linspace(
        n_sample_lower, n_sample_upper, n_sample_steps, endpoint=True
    )
    # atom_ticks = np.linspace(n_atom_lower, n_atom_upper, n_atom_steps, endpoint=True)
    atom_ticks = atom_step

    ax.set_xticks(range(n_sample_steps))
    ax.set_xticklabels([f"{int(x)}" for x in sample_ticks], fontsize=fonts)
    ax.set_yticks(range(n_atom_steps))
    ax.set_yticklabels([f"{int(x)}" for x in atom_ticks], fontsize=fonts)
    ax.set_xlabel("Number of Samples", fontsize=fonts)
    ax.set_ylabel("Number of Atoms", fontsize=fonts)

    plt.tight_layout()
    fig.savefig(figure_name, bbox_inches="tight")
    plt.close()
    print("Image " + figure_name + " has been generated.")


@click.command()
@click.option("--dataset", default="dsed", help="Choose dataset from: dsed-eq, dsed-tr, dsed, emps, whbm")
@click.option("--n_random", default=10, help="Set the number of random tests")
def main(dataset, n_random) -> None:
    match dataset:
        case "dsed-eq":
            u, y = get_dsed_eq()
            _plot_sample(
                u,
                y,
                n_sample_lower=50,
                n_sample_upper=150,
                n_sample_steps=11,
                # n_atom_lower=3,
                # n_atom_upper=120,
                # n_atom_steps=40,
                atom_step=[2, 5, 10, 15, 20, 30, 50, 70],
                figure_name="sample_dsed_eq.png",
                n_random=n_random,
                intercept=True,
                max_delay=4,
            )
        case "dsed-tr":
            u, y = get_dsed_tr()
            _plot_sample(
                u,
                y,
                n_sample_lower=50,
                n_sample_upper=250,
                n_sample_steps=11,
                # n_atom_lower=3,
                # n_atom_upper=120,
                # n_atom_steps=40,
                atom_step=[2, 5, 10, 15, 20, 30, 50, 70],
                figure_name="sample_dsed_tr.png",
                n_random=n_random,
                intercept=True,
                max_delay=6,
            )
        case "dsed":
            train_val_u, train_val_y, _ = get_dual_stable_equilibria_data()
            _plot_sample(
                train_val_u,
                train_val_y,
                n_sample_lower=50,
                n_sample_upper=250,
                n_sample_steps=11,
                # n_atom_lower=3,
                # n_atom_upper=120,
                # n_atom_steps=40,
                atom_step=[2, 5, 10, 15, 20, 30, 50, 70],
                figure_name="sample_dsed.png",
                n_random=n_random,
                intercept=True,
                max_delay=4,
            )
        case "emps":
            train_val, _ = nonlinear_benchmarks.EMPS()
            train_val_u, train_val_y = train_val
            # No intercept for EMPS dataset
            _plot_sample(
                train_val_u,
                train_val_y,
                n_sample_lower=20,
                n_sample_upper=120,
                n_sample_steps=11,
                # n_atom_lower=200,
                # n_atom_upper=1200,
                # n_atom_steps=6,
                atom_step=[2, 5, 10, 15, 20, 30, 50, 70],
                figure_name="sample_emps.png",
                n_random=n_random,
                intercept=False,
                max_delay=4,
            )
        case "whbm":
            train_val, _ = nonlinear_benchmarks.WienerHammerBenchMark()
            train_val_u, train_val_y = train_val
            _plot_sample(
                train_val_u,
                train_val_y,
                n_sample_lower=20,
                n_sample_upper=120,
                n_sample_steps=11,
                # n_atom_lower=200,
                # n_atom_upper=1200,
                # n_atom_steps=6,
                atom_step=[2, 5, 10, 15, 20, 30, 50, 70],
                figure_name="sample_whbm.png",
                n_random=n_random,
                intercept=True,
                max_delay=7,
            )
        case _:
            raise NameError(
                "The dataset is not supported. Please choose from: dsed, emps, whbm"
            )


if __name__ == "__main__":
    main()

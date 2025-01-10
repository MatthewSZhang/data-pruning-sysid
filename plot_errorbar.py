"""Generate error bar plot"""

import click
import matplotlib.pyplot as plt
import nonlinear_benchmarks
import numpy as np
from sklearn.metrics import r2_score

from utils import (
    fastcan_pruned_narx,
    get_dual_stable_equilibria_data,
    get_narx_terms,
    random_pruned_narx,
)


def _plot_errorbar(u, y, n_sample_lower, n_sample_upper, n_steps, twinx, figure_name):
    poly_terms, y, narx = get_narx_terms(u, y)

    n_random = 10
    sample_step = int((n_sample_upper - n_sample_lower) / (n_steps - 1))
    r2_fastcan = np.zeros((n_random, n_steps))
    r2_random = np.zeros((n_random, n_steps))
    for i in range(n_random):
        print(figure_name, "   ", f"Random test: {i+1}/{n_random}")
        for j in range(n_steps):
            coef, intercept = fastcan_pruned_narx(
                poly_terms, y, j * sample_step + n_sample_lower, i
            )
            r2_fastcan[i, j] = r2_score(
                np.r_[coef, intercept], np.r_[narx.coef_, narx.intercept_]
            )

            coef, intercept = random_pruned_narx(
                poly_terms, y, j * sample_step + n_sample_lower, i
            )
            r2_random[i, j] = r2_score(
                np.r_[coef, intercept], np.r_[narx.coef_, narx.intercept_]
            )

    x = np.linspace(n_sample_lower, n_sample_upper, n_steps, endpoint=True)
    fig, ax1 = plt.subplots()
    if twinx:
        ax2 = ax1.twinx()
        ax2.errorbar(
            x,
            r2_random.mean(axis=0),
            yerr=r2_random.std(axis=0) / 3,
            color="tab:orange",
            label="Random",
        )
        ax2.set_ylabel("R2")
    else:
        ax1.errorbar(
            x, r2_random.mean(axis=0), yerr=r2_random.std(axis=0) / 3, color="tab:orange", label="Random"
        )
    ax1.errorbar(
    x, r2_fastcan.mean(axis=0), yerr=r2_fastcan.std(axis=0) / 3,color="tab:blue", label="FastCan"
    )
    fig.legend(loc="upper left", bbox_to_anchor=(0.12, 0.88))
    ax1.set_ylabel("R2")
    ax1.set_xlabel("Number of Samples")
    ax1.set_xticks(x)
    fig.savefig(figure_name, bbox_inches="tight")
    plt.close()
    print("Image " + figure_name + " has been generated.")


@click.command()
@click.option("--dataset", default="dsed", help="Choose dataset from: dsed, emps, whbm")
def main(dataset) -> None:
    match dataset:
        case "dsed":
            train_val_u, train_val_y, _ = get_dual_stable_equilibria_data()
            _plot_errorbar(
                train_val_u, train_val_y, 300, 600, 11, True, "errorbar_dsed.png"
            )
        case "emps":
            train_val, _ = nonlinear_benchmarks.EMPS()
            train_val_u, train_val_y = train_val
            _plot_errorbar(
                train_val_u, train_val_y, 2000, 20000, 10, False, "errorbar_emps.png"
            )
        case "whbm":
            train_val, _ = nonlinear_benchmarks.WienerHammerBenchMark()
            train_val_u, train_val_y = train_val
            _plot_errorbar(
                train_val_u, train_val_y, 2000, 20000, 10, False, "errorbar_whbm.png"
            )
        case _:
            raise NameError(
                "The dataset is not supported. Please choose from: dsed, emps, whbm"
            )


if __name__ == "__main__":
    main()

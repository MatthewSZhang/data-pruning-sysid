"""Visualize the prediction performance of NARX models."""

import click
import matplotlib.pyplot as plt
import nonlinear_benchmarks
from fastcan.narx import make_narx
from sklearn.metrics import r2_score
import matplotlib.ticker as mticker

from utils import get_dsed_eq, get_dsed_tr, get_dual_stable_equilibria_data


def _plot_prediction(
    u,
    y,
    n_terms,
    max_delay,
    poly_degree,
    plot_n_samples,
    figure_name,
    narx=None,
    intercept=True,
):
    if narx is None:
        narx = make_narx(
            u.reshape(-1, 1),
            y,
            n_terms_to_select=n_terms,
            max_delay=max_delay,
            poly_degree=poly_degree,
            fit_intercept=intercept,
            verbose=0,
        ).fit(
            u.reshape(-1, 1),
            y,
        )
    y_hat = narx.predict(u.reshape(-1, 1), y_init=y[: narx.max_delay_])

    fonts = 15  # Define your desired font size
    plt.plot(y[:plot_n_samples], label="True")
    plt.plot(y_hat[:plot_n_samples], label="Predicted")
    # Tick label font size
    plt.tick_params(axis="both", labelsize=fonts)
    plt.xlabel("Time index", fontsize=fonts)
    plt.ylabel("Amplitude", fontsize=fonts)  # Optional, if needed
    plt.legend(fontsize=fonts, loc="upper right")
    plt.tight_layout()
    plt.title(
        f"NARX prediction results (R-squared: {
            r2_score(y[:plot_n_samples], y_hat[:plot_n_samples]):.5f})",
        fontsize=fonts,
    )
    ax = plt.gca()
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(
            lambda x, p: format(int(x), ',') if abs(int(x)) >= 10000 else str(int(x))
        )
    )
    plt.savefig(figure_name, bbox_inches="tight")
    plt.close()
    print("Image " + figure_name + " has been generated.")
    return narx


@click.command()
@click.option(
    "--dataset",
    default="dsed",
    help="Choose dataset from: dsed, emps, whbm, dsed-eq, dsed-tr",
)
@click.option("--nterms", default=10, help="Number of NARX terms (int)")
@click.option("--delay", default=10, help="Maximum delay (int)")
@click.option("--poly", default=3, help="Maximum polynomial degree of NARX terms (int)")
def main(dataset, nterms, delay, poly) -> None:
    match dataset:
        case "dsed-eq":
            train_val_u, train_val_y = get_dsed_eq()
            test_val_u1, test_val_y1, _ = get_dual_stable_equilibria_data(
                y0=[[0.6, 0.8]]
            )
            test_val_u2, test_val_y2, _ = get_dual_stable_equilibria_data(
                y0=[[-0.6, 0.8]]
            )
            narx = _plot_prediction(
                train_val_u,
                train_val_y,
                nterms,
                delay,
                poly,
                100,
                "pred_train_dsed_eq.png",
            )
            _plot_prediction(
                test_val_u1,
                test_val_y1,
                nterms,
                delay,
                poly,
                100,
                "pred_test_dsed_eq_1.png",
                narx,
            )
            _plot_prediction(
                test_val_u2,
                test_val_y2,
                nterms,
                delay,
                poly,
                100,
                "pred_test_dsed_eq_2.png",
                narx,
            )
        case "dsed-tr":
            train_val_u, train_val_y = get_dsed_tr()
            test_val_u1, test_val_y1, _ = get_dual_stable_equilibria_data(
                y0=[[0.6, 0.8]]
            )
            test_val_u2, test_val_y2, _ = get_dual_stable_equilibria_data(
                y0=[[-0.6, 0.8]]
            )
            narx = _plot_prediction(
                train_val_u,
                train_val_y,
                nterms,
                delay,
                poly,
                100,
                "pred_train_dsed_tr.png",
            )
            _plot_prediction(
                test_val_u1,
                test_val_y1,
                nterms,
                delay,
                poly,
                100,
                "pred_test_dsed_tr_1.png",
                narx,
            )
            _plot_prediction(
                test_val_u2,
                test_val_y2,
                nterms,
                delay,
                poly,
                100,
                "pred_test_dsed_tr_2.png",
                narx,
            )
        case "dsed":
            train_val_u, train_val_y, _ = get_dual_stable_equilibria_data()
            test_val_u1, test_val_y1, _ = get_dual_stable_equilibria_data(
                y0=[[0.6, 0.8]]
            )
            test_val_u2, test_val_y2, _ = get_dual_stable_equilibria_data(
                y0=[[-0.6, 0.8]]
            )
            narx = _plot_prediction(
                train_val_u,
                train_val_y,
                nterms,
                delay,
                poly,
                100,
                "pred_train_dsed.png",
            )
            _plot_prediction(
                test_val_u1,
                test_val_y1,
                nterms,
                delay,
                poly,
                100,
                "pred_test_dsed_1.png",
                narx,
            )
            _plot_prediction(
                test_val_u2,
                test_val_y2,
                nterms,
                delay,
                poly,
                100,
                "pred_test_dsed_2.png",
                narx,
            )
        case "emps":
            train_val, test_val = nonlinear_benchmarks.EMPS()
            train_val_u, train_val_y = train_val
            test_val_u, test_val_y = test_val
            narx = _plot_prediction(
                train_val_u,
                train_val_y,
                nterms,
                delay,
                poly,
                25000,
                "pred_train_emps.png",
                # intercept=False,  # No intercept for EMPS dataset
            )
            _plot_prediction(
                test_val_u,
                test_val_y,
                nterms,
                delay,
                poly,
                25000,
                "pred_test_emps.png",
                narx,
            )
        case "whbm":
            train_val, test_val = nonlinear_benchmarks.WienerHammerBenchMark()
            train_val_u, train_val_y = train_val
            test_val_u, test_val_y = test_val
            narx = _plot_prediction(
                train_val_u,
                train_val_y,
                nterms,
                delay,
                poly,
                1000,
                "pred_train_whbm.png",
            )
            _plot_prediction(
                test_val_u,
                test_val_y,
                nterms,
                delay,
                poly,
                1000,
                "pred_test_whbm.png",
                narx,
            )
        case _:
            raise NameError(
                "The dataset is not supported. Please choose from: dsed, emps, whbm"
            )


if __name__ == "__main__":
    main()

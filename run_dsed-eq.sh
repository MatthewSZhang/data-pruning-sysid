#!/bin/sh
set -e # Exit immediately if a command exits with a non-zero status

echo "Running plot_predition.py...dsed-eq...delay5"
python plot_prediction.py --dataset dsed-eq --delay 4

echo "Running plot_atom.py...dsed-eq...random10"
python plot_atom.py --dataset dsed-eq --n_random 10

echo "Running plot_errorbar.py...random10"
python plot_errorbar.py --dataset dsed-eq --n_random 10

echo "Running plot_box.py...random10"
python plot_box.py --dataset dsed-eq --n_random 10

echo "Running plot_pca.py..."
python plot_pca.py --dataset dsed-eq

echo "Running plot_batch.py...random10"
python plot_batch.py --dataset dsed-eq --n_random 10

echo "Running plot_sample.py...random10"
python plot_sample.py --dataset dsed-eq --n_random 10

echo "All scripts completed."


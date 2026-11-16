# Learning Local Differential Operators on Curves

Experimental PyTorch code for learning first- and second-order operators from local curve patches. The experiments study tangent and second-derivative directions under Euclidean, similarity, and affine transformations.

## Where to start

- `run_direction_experiment.py` trains and evaluates a model with direction-based losses. It also saves training curves and test diagnostics.
- `generate_curve_dataset.py` generates datasets of Fourier or piecewise curves.
- `models/`, `datasets/`, and `training/` contain the core model and training code.
- `utils/` contains curve generation, sampling, and geometry utilities.
- `analysis/` and the visualization folders contain experiment diagnostics and plots.

Inspect the available experiment settings with:

```bash
python run_direction_experiment.py --help
```

Generate a curve dataset with:

```bash
python generate_curve_dataset.py --output-root data/curves --family fourier
```

The direction experiment generates its own samples; it does not require the dataset-generation command above. Its options cover the transformation family, patch sampling, loss weights, and training configuration.

## Environment

The main entry points use Python, PyTorch, NumPy, and Matplotlib. Use an isolated environment with a PyTorch installation suited to your CPU or CUDA setup.

## Project status

This is an experimental research workspace. The repository includes exploratory scripts, saved outputs, and diagnostics alongside the core implementation.

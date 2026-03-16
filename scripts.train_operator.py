import numpy as np
import matplotlib.pyplot as plt

from datasets.tangent_dataset import TangentDataset


def main():

    dataset = TangentDataset(
        length=50,
        family="euclidean",

        num_curve_points=400,

        fourier_max_freq=9,
        fourier_scale=1.1,
        fourier_decay_power=1.35,

        curve_min_size=0.25,
        curve_max_size=0.95,

        patch_size=11,
        half_width=12,
        half_width_range=(8,20),

        num_negatives=10,
        negative_min_offset=4,
        negative_max_offset=70,
        negative_other_curve_fraction=0.5,

        patch_mode="jittered_symmetric",
        jitter_fraction=0.45,

        closed=True,

        point_noise_std=0.003,
        orthogonal_noise_std=0.006,

        curve_family_probs={"fourier":0.55,"piecewise":0.45},

        warp_sampling_prob=0.7,
        warp_sampling_strength=0.18,

        seed=0,
    )

    curves = []

    for i in range(50):

        rng = dataset._make_rng(i)
        curve = dataset._generate_curve(rng)

        curves.append(curve)

    fig, axes = plt.subplots(5,10,figsize=(20,10))

    for ax, curve in zip(axes.flat, curves):

        ax.plot(curve[:,0], curve[:,1], linewidth=1)

        ax.set_aspect("equal")
        ax.axis("off")

    plt.suptitle("Sample of 50 generated training curves")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

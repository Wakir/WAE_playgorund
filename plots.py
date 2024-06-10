import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt
from scipy.ndimage import gaussian_filter
from medpy.filter.smoothing import anisotropic_diffusion
from itertools import product
import os

directory = "results_gauss"
drifts = ['sudden', 'gradual']

# scales = [0.5, 0.7, 1.0]
scales = [0.5]


# chunk_classifiers = [3, 5, 10, 15]
chunk_classifiers = [15]

reference_models = ["WAE", "OOB", "UOB"]

imbalance = [0.05, 0.10, 0.20, 0.30]

ftitles = {
    "sudden_drift_5.0": "Sudden drift with 5% of minority class",
    "sudden_drift_10.0": "Sudden drift with 10% of minority class",
    "sudden_drift_20.0": "Sudden drift with 20% of minority class",
    "sudden_drift_30.0": "Sudden drift with 30% of minority class",
    "gradual_drift_5.0": "Gradual drift with 5% of minority class",
    "gradual_drift_10.0": "Gradual drift with 10% of minority class",
    "gradual_drift_20.0": "Gradual drift with 20% of minority class",
    "gradual_drift_30.0": "Gradual drift with 30% of minority class",
}

try:
    os.mkdir(os.path.join(directory, "plots"))
except:
    pass

for ind, (imb, dr) in enumerate(product(imbalance, drifts)):
    results = [np.load(os.path.join(directory, "results", f"Stream_{dr}_drift_{imb*100}_imbalance_new_wae_{scale}_{cc}_cl.npy")) for scale in scales for cc in chunk_classifiers]
    reference_results = np.concatenate([np.load(os.path.join(directory, "results", f"Stream_{dr}_drift_{imb*100}_imbalance_{model}.npy")) for model in reference_models],axis=1)
    new_results = np.concatenate((results[0], reference_results), axis=1)

    labels = [f'new_wae_{s}_{c}_cl' for s in scales for c in chunk_classifiers] + reference_models

    new_results = np.mean(new_results, axis=0)
    kernel = 1.5


    # Comparision of proposed models, one metric per figure
    fig, ax = plt.subplots(5, 1, figsize=(6.5, 9))
    metrics = ["f-score", "gmean", "bac", "precision", "recall"]
    for m_i, m in enumerate(metrics):
        colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

        ls = [
            ["-", "-", ":", ":", "--", "--", "-.", "-."],
            ["-", "-", ":", ":", ":", ":", "--", "--", "-.", "-."],
            ["-", "-", ":", ":", ":", ":", "--", "--", "-.", "-."],
        ]

        lw = [
            [1, 1, 2, 2, 1.5, 1.5, 1.5, 1.5],
            [1, 1, 1, 1, 2, 2, 1.5, 1.5, 1.5, 1.5],
            [1, 1, 1, 1, 2, 2, 1.5, 1.5, 1.5, 1.5],
        ]

        locs = [1, 3, 3]

        print(new_results.shape)

        for j, row in enumerate(new_results[:, :, m_i]):
            # res = anisotropic_diffusion(row, gamma=0.05, kappa=300, niter=10)
            res = gaussian_filter(row, kernel)
            ax[m_i].plot(
                res, c="white", ls="-",
            )
            ax[m_i].plot(
                res,
                c=colors[j],
                ls="--",
                label="%s\n$^{%.3f}$" % (labels[j], np.mean(row)),
                lw=1.0,
            )
        # ax[i].set_title(titles[i])
        ax[m_i].set_title(m)
        ax[m_i].set_ylim(0, 1)
        ax[m_i].set_xlim(0, 100 - 1)
        ax[m_i].legend(ncol=len(labels) // 2, frameon=False, fontsize=10)
        ax[m_i].spines["right"].set_color("none")
        ax[m_i].spines["top"].set_color("none")
        ax[m_i].grid(ls=":")

        # fig.suptitle(ftitles[fname] + " for " + m + " metric.")
    fig.suptitle(f"Methods comparison")
    fig.subplots_adjust(top=0.93, bottom=0.03, left=0.06, right=0.97)

    # plt.tight_layout()
    plt.savefig("foo.png")
    # plt.savefig("figures/%s.png" % (fname + "_" + m))
    plt.savefig(os.path.join(directory, "plots", "%s.png" % (f"{dr} drift {imb} imbalance_references")))
import numpy as np
import config
from scipy.stats import wilcoxon, ttest_rel, ttest_ind
from tabulate import tabulate

alpha = .05

# WHOLE
# reps x streams x methods x chunks
scores = np.load('results/res_e1_all.npy').squeeze()
print(scores.shape)

# BASE CLFS
# reps x streams x methods x chunks
scores_raw_clfs = scores[:,:,:5]
print(scores_raw_clfs.shape)

# METHODS
# reps x streams x base x criterion x borders x estimators x chunks
scores_methods = scores[:,:,5:]
scores_methods = scores_methods.reshape(10,9,5,2,10,3,499)
print(scores_methods.shape)
print("\n")

"""
Statistical analysis
"""
# Flatten chunks
# reps x streams x base x criteria x borders x estimators
results = np.mean(scores_methods, axis=6)
# Reps as last dimension
# streams x base x criteria x borders x estimators x reps
results = np.moveaxis(results, 0, 5)

# Na razie tylko SIS i DSCA
# streams x base x criteria x borders x reps
results = results[:3, :, :, :, 2]
print(results.shape)

criteria = config.criteria()
borders = config.borders()
# estimators = ["MEAN", "PREV", "DSCA"]

clfs = config.base_clf_names()
strs = config.str_weights_names()[:3]

borders_names = ["%.2f" % value for value in config.borders()]

keys = {
    0: config.criteria(),
    1: borders_names
}

# Optimas
print(results[0, 0].shape)
mresults = np.mean(results, axis=4)
for j in range(len(clfs)):
    for i in range(len(strs)):
        a = np.max(mresults[i, j])
        b = np.array(np.where(mresults[i, j] == a)).T[0]

        z = np.array(np.where(mresults[i, j] == a)).T

        print(z)

        print(
            "%5s" % clfs[j],
            " & ",
            "%12s" % strs[i],
            " & ",
            "%.3f" % a,
            " & ",
            # b,
            "\\textsc{%s}" % keys[0][b[0]],
            " & ",
            keys[1][b[1]],
            " \\\\ ",
        )

bigtables = [[]]
bigdeps = [[]]

# Iterate classifiers
for clf_idx, clf_results in enumerate(results):
    clf_name = clfs[clf_idx]

    # Iterate streams
    for s_idx, s_results in enumerate(clf_results):
        parameters = [
            "criteria",
            "borders",
        ]
        # 0 - post_pruning
        # 1 - theta
        # 2 - weight_calculation_method
        # 3 - aging_method

        # Iterate pairs of parameters
        pairs = [(0, 1)]

        for pp, pair in enumerate(pairs):
            print(
                "\n[%s-%i] %s vs %s"
                % (clf_name, s_idx, parameters[pair[0]], parameters[pair[1]])
            )
            axis = tuple([x for x in list(range(2)) if x not in pair])

            # Calculate comb results and filtered comb_results
            f_comb_results = np.mean(s_results, axis=axis)
            comb_results = np.mean(f_comb_results, axis=2)

            print(f_comb_results.shape, comb_results.shape)

            # Gather best idx
            best_idx = [i[0] for i in np.where(np.max(comb_results) == comb_results)]

            cmp_a = f_comb_results[best_idx[0], best_idx[1]]

            print(best_idx)

            dependencies = np.array(
                [
                    [
                        ttest_ind(cmp_a, f_comb_results[i, j]).pvalue
                        for j, b in enumerate(keys[pair[1]])
                    ]
                    for i, a in enumerate(keys[pair[0]])
                ]
            )
            print(dependencies)

            bigdeps[pp] += [dependencies]
            bigtables[pp] += [comb_results]

            tabres = [
                [keys[pair[0]][y_i]]
                + [
                    "%.3f %s" % (x, "d" if dependencies[y_i][x_i] >= alpha else "")
                    for x_i, x in enumerate(y)
                ]
                + ["%.3f" % np.mean(comb_results, axis=1)[y_i]]
                for y_i, y in enumerate(comb_results)
            ]

            tabres += [
                ["-mean"]
                + ["%.3f" % i for i in np.mean(comb_results, axis=0)]
                + ["%.3f" % np.mean(comb_results)]
            ]

            tab = tabulate(tabres, headers=keys[pair[1]] + ["-mean"])

            print(tab)


print("HERE")
exit()

"""
bigtables = [np.array(b) for b in bigtables]
# bigtables = [b.reshape((3, 3, b.shape[-2], b.shape[-1])) for b in bigtables]

bigdeps = [np.array(b) for b in bigdeps]
# bigdeps = [b.reshape((3, 3, b.shape[-2], b.shape[-1])) for b in bigdeps]


tables = [
    np.concatenate(
        [np.concatenate([z[i, j] for i, b in enumerate(a)]) for j, a in enumerate(z)],
        axis=1,
    )
    for z in bigtables
]


debs = [
    np.concatenate(
        [np.concatenate([z[i, j] for i, b in enumerate(a)]) for j, a in enumerate(z)],
        axis=1,
    )
    for z in bigdeps
]


pairs = combinations(list(range(4)), 2)
for i, pair in enumerate(pairs):
    print("\nTABLE %i" % i, pair, parameters[pair[0]], "vs", parameters[pair[1]])

    t = tables[i]
    d = debs[i] >= alpha

    tadam = [
        ["%s%.3f" % ("GRUBE " if d[j, i] else "", value) for i, value in enumerate(row)]
        for j, row in enumerate(t)
    ]

    grubas = tabulate(tadam, tablefmt="latex_booktabs")
    grubas = grubas.replace("GRUBE", "\\bfseries")

    f = open("tables/table_%s_%s.tex" % (parameters[pair[0]], parameters[pair[1]]), "w")
    f.write(grubas)
    f.close()

    plt.clf()
    fig, ax = plt.subplots(3, 3, sharex=True, sharey=True, figsize=(6, 5.5))

    lenj = len(keys[pair[0]])
    lenk = len(keys[pair[1]])

    if i in to_transpose:
        plt.setp(
            ax,
            xticks=range(lenj),
            xticklabels=keys[pair[0]],
            yticks=range(lenk),
            yticklabels=keys[pair[1]],
        )
    else:
        plt.setp(
            ax,
            yticks=range(lenj),
            yticklabels=keys[pair[0]],
            xticks=range(lenk),
            xticklabels=keys[pair[1]],
        )

    for j in range(3):
        for k in range(3):
            if i in to_transpose:
                smap = debs[i][
                    (j * lenk) : ((j + 1) * lenk), (k * lenj) : ((k + 1) * lenj)
                ]
                im = ax[j, k].imshow(smap, cmap="binary_r", aspect="auto")

                # Values
                for l in range(lenk):
                    for m in range(lenj):
                        ax[j, k].text(
                            m,
                            l,
                            "%.2f" % smap[l, m],
                            color="black" if smap[l, m] > 0.5 else "white",
                            ha="center",
                            va="center",
                            fontsize=9,
                        )
            else:
                smap = debs[i][
                    (k * lenj) : ((k + 1) * lenj), (j * lenk) : ((j + 1) * lenk)
                ]
                im = ax[k, j].imshow(smap, cmap="binary_r", aspect="auto")
                # Values
                print(smap, smap.shape, lenj, lenk)
                for l in range(lenj):
                    for m in range(lenk):
                        ax[k, j].text(
                            m,
                            l,
                            "%.2f" % smap[l, m],
                            color="black" if smap[l, m] > 0.5 else "white",
                            ha="center",
                            va="center",
                            fontsize=9,
                        )
            if j == 2:
                ax[j, k].set_xlabel(clfs[k])
            if k == 0:
                ax[j, k].set_ylabel(strs[j])

    fig.subplots_adjust(top=0.85, left=0.17, right=0.95, bottom=0.1)
    cbar_ax = fig.add_axes([0.17, 0.9, 0.78, 0.025])

    fig.colorbar(im, cax=cbar_ax, orientation="horizontal", ticks=[0.05, 1])

    if i in to_transpose:
        fig.suptitle(
            "%s / %s" % (parameters[pair[1]], parameters[pair[0]]), fontsize=12, x=0.57
        )
    else:
        fig.suptitle(
            "%s / %s" % (parameters[pair[0]], parameters[pair[1]]), fontsize=12, x=0.57
        )

    plt.savefig("figures/stat/p%i.png" % i)
    # plt.savefig("figures/stat/p%i.eps" % i)
"""

import numpy as np
import config
from scipy.stats import wilcoxon, ttest_rel, ttest_ind
from tabulate import tabulate
import matplotlib.pyplot as plt
from skimage.measure import label
from scipy import stats
import matplotlib.patches as patches

alpha = .05

def t_test_corrected(a, b, J=1, k=10):
    """
    Corrected t-test for repeated cross-validation.
    input, two 2d arrays. Repetitions x folds
    As default for 5x5CV
    """
    if J*k != a.shape[0]:
        raise Exception('%i scores received, but J=%i, k=%i (J*k=%i)' % (
            a.shape[0], J, k, J*k
        ))

    d = a - b
    bar_d = np.mean(d)
    bar_sigma_2 = np.var(d.reshape(-1), ddof=1)
    bar_sigma_2_mod = (1 / (J * k) + 1 / (k - 1)) * bar_sigma_2
    t_stat = bar_d / np.sqrt(bar_sigma_2_mod)
    pval = stats.t.sf(np.abs(t_stat), (k * J) - 1) * 2
    return t_stat, pval

"""
Statistical analysis
"""

# Na razie tylko SIS i DSCA
# streams x base x criteria x borders x reps
estimators = ["MEAN", "PREV", "DSCA"]
streams_names = ["SIS", "CDIS", "DDIS"]

for est_id, est_name in enumerate(estimators):
    for str_id, str_name in enumerate(streams_names):

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

        # Flatten chunks
        # reps x streams x base x criteria x borders x estimators
        results = np.mean(scores_methods, axis=6)
        # Reps as last dimension
        # streams x base x criteria x borders x estimators x reps
        results = np.moveaxis(results, 0, 5)

        # streams x base x criteria x borders x estimators x reps
        results = results[str_id*3:(str_id+1)*3, :, :, :, est_id, :]
        print(results.shape)

        criteria = config.criteria()
        borders = config.borders()

        clfs = config.base_clf_names()

        # STREAM
        strs = config.str_weights_names()[str_id*3:(str_id+1)*3]

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
        for clf_idx in range(len(clfs)):
            clf_results = results[:, clf_idx]
            clf_name = clfs[clf_idx]

            # Iterate streams
            for s_idx, s_results in enumerate(clf_results):
                parameters = [
                    "criteria",
                    "borders",
                ]

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
        # if est_name == "PREV" and str_name == "CDIS":
        #     exit()
        print("HERE")
        # exit()
        bigtables = [np.array(b) for b in bigtables]
        print(bigtables[0].shape)
        print(bigtables[0][5,0])
        bigtables = [b.reshape((5, 3, b.shape[-2], b.shape[-1])) for b in bigtables]
        bigtables[0] = np.swapaxes(bigtables[0], 0, 1)
        # print(clfs)
        # print(bigtables[0].shape)
        # print(bigtables[0][2,1,0])
        # exit()

        bigdeps = [np.array(b) for b in bigdeps]
        bigdeps = [b.reshape((5, 3, b.shape[-2], b.shape[-1])) for b in bigdeps]
        bigdeps[0] = np.swapaxes(bigdeps[0], 0, 1)

        tables = [
            np.concatenate(
                [np.concatenate([z[j, i] for i, b in enumerate(a)]) for j, a in enumerate(z)],
                axis=1,
            )
            for z in bigtables
        ]


        debs = [
            np.concatenate(
                [np.concatenate([z[j, i] for i, b in enumerate(a)]) for j, a in enumerate(z)],
                axis=1,
            )
            for z in bigdeps
        ]

        # pairs = combinations(list(range(4)), 2)
        pairs = [(0, 1)]
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

            # f = open("tables/table_%s_%s.tex" % (parameters[pair[0]], parameters[pair[1]]), "w")
            # f.write(grubas)
            # f.close()

            plt.clf()
            fig, ax = plt.subplots(3, 5, sharex=True, sharey=True, figsize=(6, 6))

            lenj = len(keys[pair[0]])
            lenk = len(keys[pair[1]])
            print(keys[pair[1]])

            plt.setp(
                ax,
                xticks=range(lenj),
                xticklabels=keys[pair[0]],
                yticks=range(lenk),
                yticklabels=keys[pair[1]],
            )

            for j in range(3):
                for k in range(5):
                    # ax[j, k].xaxis.set_tick_params(rotation=-60)
                    bacmap = tables[i][
                        (k * lenj) : ((k + 1) * lenj), (j * lenk) : ((j + 1) * lenk)
                    ]
                    smap = debs[i][
                        (k * lenj) : ((k + 1) * lenj), (j * lenk) : ((j + 1) * lenk)
                    ]

                    im = ax[j, k].imshow(smap.T, cmap="binary_r", aspect="auto")

                    # Values
                    smap = np.around(smap, 3)
                    smapunique = np.unique(smap)
                    saddress = np.array(np.meshgrid(*[range(_) for _ in smap.shape]))

                    for sunique in smapunique:
                        mask = smap.T == sunique

                        regions = label(mask, connectivity=1)

                        for sub_idx in range(np.max(regions)):
                            print('S', sub_idx)
                            submask = regions == (sub_idx+1)
                            print(submask)

                            vvvvv = np.mean(bacmap[submask.T])
                            vvvvv = sunique

                            aa = saddress[0,:,:][submask]
                            bb = saddress[1,:,:][submask]
                            a = np.mean(aa, axis=0)
                            b = np.mean(bb, axis=0)
                            #print(saddress[0,:,:])

                            rect = patches.Rectangle((np.min(aa)-.25,
                                                      np.min(bb)-.25),
                                                     np.max(aa)-np.min(aa)+.5,
                                                     np.max(bb)-np.min(bb)+.5, linewidth=1,
                                                     edgecolor='tomato',
                                                     ls=':', facecolor='none')

                            # Add the patch to the Axes
                            ax[j,k].add_patch(rect)

                            print(a)
                            print(b)
                            ax[j, k].text(
                                a,
                                b,
                                "%.3f" % vvvvv,
                                color="black" if sunique > np.mean(smap) else "white",
                                ha="center",
                                va="center",
                                fontsize=9,
                                # rotation=90
                            )

                    for l in range(lenj):
                        for m in range(lenk):
                            '''
                            ax[j, k].text(
                                l,
                                m,
                                "%.3f" % bacmap[l, m],
                                color="black" if smap[l, m] > np.mean(smap) else "white",
                                ha="center",
                                va="center",
                                fontsize=9,
                                # rotation=90
                            )
                            '''
                            pass
                    if j == 2:
                        ax[j, k].set_xlabel(clfs[k])
                    if k == 0:
                        ax[j, k].set_ylabel(strs[j])

            fig.subplots_adjust(top=0.85, left=0.12, right=0.95, bottom=0.1)
            cbar_ax = fig.add_axes([0.1, 0.9, 0.85, 0.025])

            fig.colorbar(im, cax=cbar_ax, orientation="horizontal", ticks=[0.05, 1])

            # fig.suptitle(
            #     "%s / %s / %s / %s" % (est_name, str_name, parameters[pair[0]], parameters[pair[1]]), fontsize=12, x=0.57
            # )
            fig.suptitle(
                "%s | %s" % (est_name, str_name), fontsize=12, x=0.5
            )
            #plt.tight_layout()

            plt.savefig('foo.png')
            plt.savefig("figures/stat/%s_%s.png" % (est_name, str_name), dpi=200)
            plt.savefig("figures/stat/%s_%s.eps" % (est_name, str_name), dpi=200)

            plt.close()
            # plt.savefig("figures/stat/p%i.eps" % i)

            # exit()

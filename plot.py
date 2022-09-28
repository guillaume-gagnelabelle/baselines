import argparse
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

args_form = argparse.ArgumentParser(allow_abbrev=False)
args_form.add_argument("--s", type=int, default=0)
args_form.add_argument("--e", type=int, default=-1)
args_form.add_argument("--dataset", type=str, default="mnist", choices=["cifar", "mnist"])
args_form.add_argument("--baseline", type=str, default="pham", choices=["fixmatch", "simclr", "pham", "mt"])

num_labeled = [3000, 1500, 900, 600, 300]
seeds = [0, 1, 2, 3, 4]

args_orig = args_form.parse_args()
if args_orig.e == -1: args_orig.e = args_orig.s

nb_graphs = 9
fig, axarr = plt.subplots(nb_graphs, figsize=(5, nb_graphs * 3))
fname = [["./results/%s/%s_%s@%s_%s.pytorch" % (args_orig.dataset, args_orig.baseline, args_orig.dataset, nb_lbl, str(seed)) for nb_lbl in num_labeled] for seed in seeds]
for k in range(len(num_labeled)):
    for s in range(len(seeds)):
        logs = torch.load(fname[s][k], map_location=device)
        args = logs["args.logs"]

        metrics = list(args.keys())
        if s == args_orig.s:
            titles = metrics
            xs = [[[] for _ in range(len(metrics))] for _ in range(len(num_labeled))]
            ys = [[[] for _ in range(len(metrics))] for _ in range(len(num_labeled))]

        for i, metric in enumerate(metrics):
            xs[k][i].append(list(args[metric].keys()))
            ys[k][i].append(list(args[metric].values()))
        print(metrics)
        print(args.keys())


    for i in range(len(xs[k])):
        # if 'val_train_loss' in titles[i]: continue
        # if 'val_test_loss' in titles[i]: continue

        x = np.array(xs[k][i][0])
        y = np.array(ys[k][i])

        y_mean = y.mean(axis=0)
        y_std = y.std(axis=0)

        axarr[i].set_title(titles[i])
        axarr[i].plot(x, y_mean, label=str(num_labeled[k]))
        axarr[i].fill_between(x, y_mean - y_std, y_mean + y_std, alpha=0.5)
        axarr[i].legend()

savepath = "./results/%s/plots_%s_%s.png" % (args_orig.dataset, args_orig.baseline, args_orig.dataset)

plt.tight_layout()
fig.savefig(savepath)
plt.close("all")

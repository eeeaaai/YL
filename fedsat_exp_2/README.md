
# FedSat/FedAsync/FedAvg Minimal Reproduction (with CSV logging)

## Plotting test accuracy vs. time
Log evaluations to CSV using `--log_csv`, then plot with `plot_acc.py`:

```bash
# Example: run two scenarios and log to separate CSVs
python -m fedsat_exp.train --dataset mnist --model logreg --algo fedsat --preset bremen --horizon_min 300 --local_steps 50 --lr 0.1 --log_csv fedsat_mnist.csv --run_name bremen_nonIID

python -m fedsat_exp.train --dataset cifar10 --model resnet18 --algo fedasync --preset pole --iid --horizon_min 300 --local_steps 100 --lr 0.1 --alpha_base 0.1 --eps 0.1 --a 1.0 --To_max_min 127 --log_csv fedasync_cifar.csv --run_name pole_IID

# Make a single plot with both curves
python -m fedsat_exp.plot_acc fedsat_mnist.csv fedasync_cifar.csv --out acc_vs_time.png
```

The CSV columns are:
`t_min, acc, algo, preset, model, dataset, seed, local_steps, lr, lam, alpha_base, eps, a, To_max_min, run_name`.
```

# rest of the original README omitted for brevity.

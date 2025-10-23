
import argparse, random
import torch
import torch.nn.functional as F

from utils import get_device
from simulator import EventLoop, preset_bremen_two_shells, preset_pole_single_shell
from algorithms import fedavg_aggregate, fedasync_update, fedsat_update
from data import get_mnist_loaders, get_cifar10_loaders, iid_partition, noniid_partition_by_label, make_client_loaders, make_test_loader
from models import LogisticMNIST, resnet18_cifar10, TinySNN_MNIST, TinySNN_CIFAR10

ALGOS = ["fedavg_sync", "fedasync", "fedsat"]

def evaluate(model, test_loader, device):
    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            preds = logits.argmax(dim=1)
            total += yb.numel()
            correct += (preds == yb).sum().item()
    return correct / max(total,1)

def satellite_local_train(model, theta_global, loader, steps, lr, lam, device):
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    it = 0
    model.train()
    theta_prime = {k: v.detach().clone() for k, v in theta_global.items()}
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        logits = model(xb)
        loss = F.cross_entropy(logits, yb)
        if lam > 0:
            reg = 0.0
            for (n, p) in model.named_parameters():
                reg = reg + (p - theta_prime[n]).pow(2).sum()
            loss = loss + 0.5*lam*reg
        loss.backward()
        opt.step()
        it += 1
        if it >= steps: break
    return {k: v.detach().clone() for k, v in model.state_dict().items()}

def build_model(dataset, model_name):
    if dataset == "mnist":
        if model_name == "logreg":
            return LogisticMNIST()
        elif model_name == "snn":
            return TinySNN_MNIST(steps=20, leak=0.95)
        else:
            raise ValueError("ResNet18 not supported for MNIST in this template.")
    elif dataset == "cifar10":
        if model_name == "resnet18":
            return resnet18_cifar10()
        elif model_name == "snn":
            return TinySNN_CIFAR10(steps=20, leak=0.95)
        else:
            raise ValueError("LogReg not supported for CIFAR-10; choose resnet18 or snn.")
    else:
        raise ValueError("Unknown dataset")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["mnist","cifar10"], default="mnist")
    ap.add_argument("--model", choices=["logreg","resnet18","snn"], default="snn")
    ap.add_argument("--algo", choices=ALGOS, default="fedsat")
    ap.add_argument("--preset", choices=["bremen","pole"], default="bremen")
    ap.add_argument("--iid", action="store_true", help="Use IID split (else non-IID).")
    ap.add_argument("--horizon_min", type=float, default=600.0)
    ap.add_argument("--local_steps", type=int, default=50)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--lam", type=float, default=0.0)
    ap.add_argument("--alpha_base", type=float, default=0.1)
    ap.add_argument("--eps", type=float, default=0.1)
    ap.add_argument("--a", type=float, default=1.0)
    ap.add_argument("--To_max_min", type=float, default=127.0)
    ap.add_argument("--batch_size", type=int, default=10)
    ap.add_argument("--eval_every_min", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log_csv", type=str, default="1", help="Append eval rows to this CSV if set.")
    ap.add_argument("--run_name", type=str, default="", help="Optional run tag to store in CSV.")
    args = ap.parse_args()

    random.seed(args.seed); torch.manual_seed(args.seed)
    device = get_device()
    print(f"Device: {device}")

    # Data
    if args.dataset == "mnist":
        trainset, testset = get_mnist_loaders(root="./data")
    else:
        trainset, testset = get_cifar10_loaders(root="./data")

    # Global model
    model = build_model(args.dataset, args.model).to(device)
    theta_global = {k: v.detach().clone() for k, v in model.state_dict().items()}

    # Partition
    n_clients = 10 if args.preset=="bremen" else 6
    parts = iid_partition(trainset, n_clients) if args.iid else noniid_partition_by_label(trainset, n_clients, 2)
    client_loaders = make_client_loaders(trainset, parts, batch_size=args.batch_size, shuffle=True)
    test_loader = make_test_loader(testset, batch_size=128)

    # Contact presets
    sats = (preset_bremen_two_shells() if args.preset=="bremen"
            else preset_pole_single_shell())

    last_pull_time = {s.sid: 0.0 for s in sats}
    last_pulled_theta = {s.sid: {k: v.detach().clone() for k, v in theta_global.items()} for s in sats}
    n_total = sum(s.local_data_size for s in sats)

    next_eval_t = args.eval_every_min * 60.0

    # Per-satellite models (same architecture) without re-importing
    sat_models = {}
    for s in sats:
        sm = build_model(args.dataset, args.model).to(device)
        sm.load_state_dict(theta_global)
        sat_models[s.sid] = sm

    sat_loader = {sid: client_loaders[i] for i, sid in enumerate([s.sid for s in sats])}
    pending = []
    loop = EventLoop(sats, horizon_min=args.horizon_min)
    print(f"[dbg] queued events: {len(loop.pq)}")


    def on_event(t, etype, sid, k_pass):
        nonlocal theta_global, next_eval_t, pending
        s = [x for x in sats if x.sid==sid][0]
        if etype == "contact_start":
            sat_models[sid].load_state_dict(theta_global)
            last_pull_time[sid] = t
            theta_local = satellite_local_train(sat_models[sid], theta_global, sat_loader[sid],
                                                steps=args.local_steps, lr=args.lr, lam=args.lam, device=device)
            sat_models[sid].local_buffer = theta_local
        elif etype == "contact_end":
            theta_local = sat_models[sid].local_buffer
            if args.algo == "fedavg_sync":
                pending.append((theta_local, s.local_data_size))
            elif args.algo == "fedasync":
                delta_t = t - last_pull_time[sid]
                alpha, st = fedasync_update(theta_global, theta_local, delta_t,
                                            args.alpha_base, args.eps, args.To_max_min, args.a)
            elif args.algo == "fedsat":
                fedsat_update(theta_global, last_pulled_theta[sid], theta_local,
                              nk=s.local_data_size, n_total=n_total)
            last_pulled_theta[sid] = {k: v.detach().clone() for k, v in theta_global.items()}

        if t >= next_eval_t:
            mtmp = build_model(args.dataset, args.model).to(device)
            mtmp.load_state_dict(theta_global)
            acc = evaluate(mtmp, test_loader, device)
            print(f"[t={t/60.0:.1f} min] test acc={acc*100:.2f}% algo={args.algo}")
            if args.log_csv:
                import csv, os
                header = ["t_min","acc","algo","preset","model","dataset","seed","local_steps","lr","lam","alpha_base","eps","a","To_max_min","run_name"]
                row = [t/60.0, acc, args.algo, args.preset, args.model, args.dataset, args.seed,
                       args.local_steps, args.lr, args.lam, args.alpha_base, args.eps, args.a, args.To_max_min, args.run_name]
                file_exists = os.path.exists(args.log_csv)
                with open(args.log_csv, "a", newline="") as f:
                    w = csv.writer(f)
                    if not file_exists:
                        w.writerow(header)
                    w.writerow(row)
            next_eval_t += args.eval_every_min * 60.0

        if args.algo == "fedavg_sync" and len(pending) >= len(sats):
            local_thetas, weights = zip(*pending)
            fedavg_aggregate(local_thetas, weights, theta_global)
            pending = []
    loop.run(on_event)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  6 18:32:18 2025

@author: egemenalacali
"""

import argparse, random, math
import torch
import torch.nn.functional as F

from utils import get_device
from simulator import EventLoop, preset_bremen_two_shells, preset_pole_single_shell
from algorithms import fedavg_aggregate, fedasync_update, fedsat_update
from data import (
    get_mnist_loaders, get_cifar10_loaders,
    iid_partition, noniid_partition_by_label,
    make_client_loaders, make_test_loader
)
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
        if it >= steps:
            break
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
    ap.add_argument("--model", choices=["logreg","resnet18","snn"], default="logreg")
    ap.add_argument("--algo", choices=ALGOS, default="fedsat")
    ap.add_argument("--preset", choices=["bremen","pole"], default="bremen")
    ap.add_argument("--iid", action="store_true", help="Use IID split (equal sizes).")
    ap.add_argument("--horizon_min", type=float, default=600.0)
    ap.add_argument("--local_steps", type=int, default=50)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--lam", type=float, default=0.0)
    ap.add_argument("--alpha_base", type=float, default=0.1)
    ap.add_argument("--eps", type=float, default=0.01)
    ap.add_argument("--a", type=float, default=640.35)  # ≈ 5*(1+eps)*127
    ap.add_argument("--To_max_min", type=float, default=127.0)
    ap.add_argument("--batch_size", type=int, default=10)
    ap.add_argument("--eval_every_min", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log_csv", type=str, default="", help="Append eval rows to this CSV if set.")
    ap.add_argument("--run_name", type=str, default="", help="Optional run tag to store in CSV.")
    # GPU/Loader knobs
    ap.add_argument("--workers", type=int, default=2, help="DataLoader workers (Colab: 2 is safe).")
    ap.add_argument("--amp", action="store_true", help="Enable mixed precision (recommended on GPU).")
    ap.add_argument("--pin_memory", action="store_true", help="Pin host memory for faster GPU transfer.")
    # Asynchrony realism
    ap.add_argument("--train_between_passes", action="store_true", help="Train between passes; upload next pass end.")
    ap.add_argument("--one_epoch_per_gap", action="store_true", help="Use exactly one pass over local shard between contacts.")
    ap.add_argument("--sec_per_step", type=float, default=0.0, help="If >0, cap steps by floor(available_seconds/sec_per_step).")
    ap.add_argument("--vis_stride_shell0", type=int, default=1)
    ap.add_argument("--vis_stride_shell1", type=int, default=1)
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
    client_loaders = make_client_loaders(trainset, parts, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    test_loader = make_test_loader(testset, batch_size=128, num_workers=args.workers)

    # Contact presets
    sats = (preset_bremen_two_shells() if args.preset=="bremen" else preset_pole_single_shell())
    sid2sat = {s.sid: s for s in sats}

    last_pull_time = {s.sid: 0.0 for s in sats}
    last_pulled_theta = {s.sid: {k: v.detach().clone() for k, v in theta_global.items()} for s in sats}
    last_contact_end = {s.sid: None for s in sats}   # for budgeting on next cycle
    n_total = sum(s.local_data_size for s in sats)

    next_eval_t = args.eval_every_min * 60.0

    # Per-satellite models (same architecture)
    sat_models = {}
    for s in sats:
        sm = build_model(args.dataset, args.model).to(device)
        sm.load_state_dict(theta_global)
        # runtime flags on the object
        sm.will_train = False
        sm.local_buffer = None
        sat_models[s.sid] = sm

    sat_loader = {sid: client_loaders[i] for i, sid in enumerate([s.sid for s in sats])}
    pending_updates = {}  # sid -> (state_dict, weight)
    def try_sync_aggregate():
        nonlocal theta_global, pending_updates, last_pulled_theta
        if len(pending_updates) == len(sats):
            local_thetas, weights = zip(*pending_updates.values())
            fedavg_aggregate(local_thetas, weights, theta_global)
            # after new global, refresh "prevpull" snapshots
            for s in sats:
                last_pulled_theta[s.sid] = {k: v.detach().clone() for k, v in theta_global.items()}
            pending_updates.clear()
    # --------------
    next_eval_t = args.eval_every_min * 60.0
    loop = EventLoop(sats, horizon_min=args.horizon_min)
    print(f"[dbg] queued events: {len(loop.pq)}")

    def budgeted_steps(sid:int, now_t:float, mode:str):
        """
        mode: 'between' or 'contact'
        - 'between': budget = (time since last_contact_end) if available, else (period - contact_duration)
        - 'contact' : budget = contact_duration
        """
        s = sid2sat[sid]
        if args.sec_per_step <= 0:
            return args.local_steps  # no budgeting

        if mode == "between":
            # prefer exact time since last end if we have it
            if last_contact_end[sid] is not None:
                budget_sec = max(0.0, now_t - last_contact_end[sid])
            else:
                budget_sec = max(0.0, (s.period_min - s.contact_duration_min) * 60.0)
        else:  # 'contact'
            budget_sec = max(0.0, s.contact_duration_min * 60.0)

        max_steps = int(budget_sec // args.sec_per_step)
        return max(0, min(args.local_steps, max_steps))

    def on_event(t, etype, sid, k_pass):
        nonlocal theta_global, next_eval_t
        s = sid2sat[sid]

        # -------- TRAIN-BETWEEN-PASSES MODE --------
        if args.train_between_passes:
            if etype == "contact_start":
                # 1) If we owed an upload from training that happened between previous end and this start,
                #    finish that training now (instantly in sim) and buffer it.
                if getattr(sat_models[sid], "will_train", False):
                    steps = budgeted_steps(sid, t, mode="between")
                    if steps > 0:
                        theta_local = satellite_local_train(
                            sat_models[sid], last_pulled_theta[sid], sat_loader[sid],
                            steps=steps, lr=args.lr, lam=args.lam, device=device
                        )
                        sat_models[sid].local_buffer = theta_local
                    sat_models[sid].will_train = False

                # 2) Pull current global to start a NEW between-pass training cycle
                sat_models[sid].load_state_dict(theta_global)
                last_pull_time[sid] = t
                last_pulled_theta[sid] = {k: v.detach().clone() for k, v in theta_global.items()}
                # mark that we will train until next contact_start
                sat_models[sid].will_train = True

            elif etype == "contact_end":
                # Upload if we have a buffered update (i.e., from previous between-pass training)
                if sat_models[sid].local_buffer is not None:
                    theta_local = sat_models[sid].local_buffer
                    if args.algo == "fedavg_sync":
                        pending_updates[sid] = (theta_local, s.local_data_size)
                        try_sync_aggregate()
                    elif args.algo == "fedasync":
                        delta_t = t - last_pull_time[sid]
                        _alpha, _st = fedasync_update(
                            theta_global, theta_local, delta_t,
                            args.alpha_base, args.eps, args.To_max_min, args.a
                        )
                    elif args.algo == "fedsat":
                        fedsat_update(
                            theta_global, last_pulled_theta[sid], theta_local,
                            nk=s.local_data_size, n_total=n_total
                        )
                    # clear buffer after upload
                    sat_models[sid].local_buffer = None

                # update last end time for budgeting the next gap
                last_contact_end[sid] = t

        # -------- ORIGINAL (TRAIN-DURING-CONTACT) MODE --------
        else:
            if etype == "contact_start":
                # pull now
                sat_models[sid].load_state_dict(theta_global)
                last_pull_time[sid] = t
                # train instantly but budget by contact duration if sec_per_step>0
                steps = budgeted_steps(sid, t, mode="contact")
                if steps > 0:
                    theta_local = satellite_local_train(
                        sat_models[sid], theta_global, sat_loader[sid],
                        steps=steps, lr=args.lr, lam=args.lam, device=device
                    )
                else:
                    # do nothing (no time) -> treat as identity update
                    theta_local = {k: v.detach().clone() for k, v in sat_models[sid].state_dict().items()}
                sat_models[sid].local_buffer = theta_local

            elif etype == "contact_end":
                theta_local = sat_models[sid].local_buffer
                if args.algo == "fedavg_sync":
                        pending_updates[sid] = (theta_local, s.local_data_size)
                        try_sync_aggregate()
                elif args.algo == "fedasync":
                    delta_t = t - last_pull_time[sid]
                    _alpha, _st = fedasync_update(
                        theta_global, theta_local, delta_t,
                        args.alpha_base, args.eps, args.To_max_min, args.a
                    )
                elif args.algo == "fedsat":
                    fedsat_update(
                        theta_global, last_pulled_theta[sid], theta_local,
                        nk=s.local_data_size, n_total=n_total
                    )
                last_pulled_theta[sid] = {k: v.detach().clone() for k, v in theta_global.items()}
                last_contact_end[sid] = t

        # -------- Periodic evaluation & CSV logging --------
        if t >= next_eval_t:
            mtmp = build_model(args.dataset, args.model).to(device)
            mtmp.load_state_dict(theta_global)
            acc = evaluate(mtmp, test_loader, device)
            print(f"[t={t/60.0:.1f} min] test acc={acc*100:.2f}% algo={args.algo}")
            if args.log_csv:
                import csv, os
                header = ["t_min","acc","algo","preset","model","dataset","seed",
                          "local_steps","lr","lam","alpha_base","eps","a","To_max_min","run_name",
                          "train_between_passes","sec_per_step"]
                row = [t/60.0, acc, args.algo, args.preset, args.model, args.dataset, args.seed,
                       args.local_steps, args.lr, args.lam, args.alpha_base, args.eps, args.a, args.To_max_min,
                       args.run_name, int(args.train_between_passes), args.sec_per_step]
                file_exists = os.path.exists(args.log_csv)
                with open(args.log_csv, "a", newline="") as f:
                    w = csv.writer(f)
                    if not file_exists:
                        w.writerow(header)
                    w.writerow(row)
            next_eval_t += args.eval_every_min * 60.0

        # FedAvg sync aggregation whenever we've heard from everyone in a "cycle"
        if args.algo == "fedavg_sync" and len(pending) >= len(sats):
            local_thetas, weights = zip(*pending)
            fedavg_aggregate(local_thetas, weights, theta_global)
            pending = []

    # RUN THE EVENT LOOP
    loop.run(on_event)

    # Always print a final evaluation at horizon end
    mtmp = build_model(args.dataset, args.model).to(device)
    mtmp.load_state_dict(theta_global)
    final_acc = evaluate(mtmp, test_loader, device)
    print(f"[final t={args.horizon_min:.1f} min] test acc={final_acc*100:.2f}% algo={args.algo}")

if __name__ == "__main__":
    main()

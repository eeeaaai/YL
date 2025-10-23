#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 23 05:07:12 2025

@author: egemenalacali

Notes:
- Evaluation modes: time / updates / contact_end
- Event logging: contact_start + uploaded update lines printed next to eval
- This version enforces (optionally) half-class-per-shell distributions for two-shell presets
  via --shell_half_classes (shell 0 -> classes [0..4], shell 1 -> classes [5..9]).
"""

import argparse, random, math
import torch
import torch.nn.functional as F
import os, json, datetime

from matplotlib import pyplot as plt

from utils import get_device
from simulator import EventLoop, preset_bremen_two_shells, preset_pole_single_shell
from algorithms import fedavg_aggregate, fedasync_update, fedsat_update
from data import (
    get_mnist_loaders, get_cifar10_loaders,
    iid_partition, noniid_partition_by_label,
    make_client_loaders, make_test_loader
)
from models import LogisticMNIST, resnet18_cifar10, TinySNN_MNIST, TinySNN_CIFAR10
from collections import Counter
import numpy as np

ALGOS = ["fedavg_sync", "fedasync", "fedsat"]

# -------------------- Utils for labels / shell-restricted partition --------------------

def _get_targets_vector(dataset):
    # Handles common torchvision datasets (MNIST/CIFAR10)
    if hasattr(dataset, "targets"):
        t = dataset.targets
        if isinstance(t, list):
            import torch
            t = torch.tensor(t)
        return t
    if hasattr(dataset, "train_labels"):  # old torchvision
        return dataset.train_labels
    raise AttributeError("Cannot locate labels/targets on dataset")

def _indices_for_labels(targets, allowed_labels):
    allowed = set(int(x) for x in allowed_labels)
    return [i for i, y in enumerate(targets) if int(y) in allowed]

def _split_equal_chunks(indices, k):
    # Deterministic equal split
    n = len(indices)
    chunk_sizes = [n // k + (1 if r < (n % k) else 0) for r in range(k)]
    out, start = [], 0
    for cs in chunk_sizes:
        out.append(indices[start:start+cs])
        start += cs
    return out

def _shell_half_class_parts(trainset, sats, iid, labels_per_client=2, seed=0):
    """
    Build a list-of-index-lists (parts), aligned with 'sats' order, such that:
      - Shell 0 clients receive only classes [0..4]
      - Shell 1 clients receive only classes [5..9]
    If iid=True -> equal split of all indices in each shell's allowed class set.
    If iid=False -> non-IID by label: each client in the shell gets 'labels_per_client' labels
                    sampled without replacement (cycling if needed).
    """
    import random as pyrand
    pyrand.seed(seed)

    targets = _get_targets_vector(trainset)
    # Default to 10-class assumption (MNIST/CIFAR10). You can generalize if needed.
    classes_shell0 = list(range(0, 5))
    classes_shell1 = list(range(5, 10))

    # Group satellites by shell attribute (assume attribute 'shell' exists; fallback 0)
    shell_to_sids = {0: [], 1: []}
    for s in sats:
        sh = getattr(s, "shell", 0)
        sh = 0 if sh not in (0,1) else sh
        shell_to_sids[sh].append(s.sid)

    # Build parts aligned with 'sats' order
    sid_order = [s.sid for s in sats]
    sid_to_part = {sid: [] for sid in sid_order}

    for shell_id, sids in shell_to_sids.items():
        if not sids:
            continue
        allowed = classes_shell0 if shell_id == 0 else classes_shell1
        shell_indices = _indices_for_labels(targets, allowed)
        if iid:
            # Shuffle once for determinism by seed
            rng = torch.Generator().manual_seed(seed + shell_id)
            perm = torch.randperm(len(shell_indices), generator=rng).tolist()
            shell_indices = [shell_indices[i] for i in perm]
            chunks = _split_equal_chunks(shell_indices, len(sids))
            for sid, chunk in zip(sids, chunks):
                sid_to_part[sid] = chunk
        else:
            # non-IID by label inside the shell: each client gets 'labels_per_client' labels
            # Build per-label index pools
            label_to_idxs = {c: [] for c in allowed}
            for i, y in enumerate(targets):
                y = int(y)
                if y in label_to_idxs:
                    label_to_idxs[y].append(i)
            # Shuffle each label pool deterministically
            for j, c in enumerate(allowed):
                rng = torch.Generator().manual_seed(seed + 100*shell_id + j)
                idxs = label_to_idxs[c]
                if len(idxs) > 0:
                    perm = torch.randperm(len(idxs), generator=rng).tolist()
                    label_to_idxs[c] = [idxs[p] for p in perm]
            # Assign labels to clients (cycling through allowed set)
            L = len(allowed)
            for k, sid in enumerate(sids):
                chosen = [allowed[(k*labels_per_client + j) % L] for j in range(labels_per_client)]
                bucket = []
                for c in chosen:
                    take = max(1, len(label_to_idxs[c]) // max(1, len(sids)))  # roughly equal take
                    bucket.extend(label_to_idxs[c][:take])
                    label_to_idxs[c] = label_to_idxs[c][take:]
                sid_to_part[sid] = bucket

    # Return parts aligned to 'sats' order
    parts = [sid_to_part[sid] for sid in sid_order]
    return parts

# --------------------------------------------------------------------------------------

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
    # reference snapshot for proximal regularization
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

def make_run_slug(args):
    bits = [
        f"algo={args.algo}",
        f"preset={args.preset}",
        f"model={args.model}",
        f"data={args.dataset}",
        f"iid={int(bool(args.iid))}",
        f"shellHalf={int(bool(args.shell_half_classes))}",
        f"steps={args.local_steps}",
        f"bs={args.batch_size}",
        f"lr={args.lr:g}",
        f"lam={args.lam:g}",
        f"seed={args.seed}",
        f"emode={args.eval_mode}",
    ]
    if args.run_name:
        bits.append(f"run={args.run_name}")
    # safe file slug
    slug = "__".join(bits).replace("/", "-")
    return slug



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["mnist","cifar10"], default="mnist")
    ap.add_argument("--model", choices=["logreg","resnet18","snn"], default="logreg")
    ap.add_argument("--algo", choices=ALGOS, default="fedasync")
    ap.add_argument("--preset", choices=["bremen","pole"], default="bremen")
    ap.add_argument("--iid", action="store_false", help="Use IID split (equal sizes).")
    ap.add_argument("--horizon_min", type=float, default=1440.0)
    ap.add_argument("--local_steps", type=int, default=50)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--lam", type=float, default=0.0)
    ap.add_argument("--alpha_base", type=float, default=0.1)
    ap.add_argument("--eps", type=float, default=0.01)
    ap.add_argument("--a", type=float, default=640.35)  # ≈ 5*(1+eps)*127
    ap.add_argument("--To_max_min", type=float, default=127.0)
    ap.add_argument("--batch_size", type=int, default=10)
    ap.add_argument("--eval_every_min", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run_name", type=str, default="", help="Optional run tag for plot title.")
    # GPU/Loader knobs
    ap.add_argument("--workers", type=int, default=2, help="DataLoader workers (Colab: 2 is safe).")
    ap.add_argument("--amp", action="store_true", help="Enable mixed precision (recommended on GPU).")
    ap.add_argument("--pin_memory", action="store_true", help="Pin host memory for faster GPU transfer.")
    # Asynchrony realism
    ap.add_argument("--train_between_passes", action="store_false", help="Train between passes; upload next pass end.")
    ap.add_argument("--one_epoch_per_gap", action="store_false", help="Use exactly one pass over local shard between contacts.")
    ap.add_argument("--sec_per_step", type=float, default=0.0, help="If >0, cap steps by floor(available_seconds/sec_per_step).")
    ap.add_argument("--vis_stride_shell0", type=int, default=1)
    ap.add_argument("--vis_stride_shell1", type=int, default=1)

    # NEW: enforce half-class-per-shell split for two-shell presets
    ap.add_argument("--shell_half_classes", action="store_false",
                    help="In two-shell presets, restrict shell 0 to classes [0-4] and shell 1 to [5-9].")

    ap.add_argument("--eval_mode", choices=["time","updates","contact_end"], default="time",
                help="time: every eval_every_min minutes; updates: every K global applies; contact_end: after each upload is applied")
    ap.add_argument("--eval_stride_updates", type=int, default=1,
                help="Evaluate every K applied global updates when --eval_mode=updates")
    ap.add_argument("--out_dir", type=str, default="runs",
                help="Directory to save CSV/plots for this run.")
    ap.add_argument("--p_comm_success", type=float, default=0.80,
                help="Per-pass success probability for GS contact (0..1). If a pass fails, skip both download and upload.")



    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    run_slug = make_run_slug(args)
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    
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
    print("Classifier out_features:", model.fc.out_features)  # or model.classifier[-1].out_features / model.linear.out_features

    theta_global = {k: v.detach().clone() for k, v in model.state_dict().items()}

    # Contact presets
    sats = (preset_bremen_two_shells() if args.preset=="bremen" else preset_pole_single_shell())
    sid2sat = {s.sid: s for s in sats}

    # Partition (possibly shell-restricted)
    if args.shell_half_classes and args.preset == "bremen":
        # Restrict per shell
        parts = _shell_half_class_parts(
            trainset=trainset, sats=sats,
            iid=args.iid, labels_per_client=2, seed=args.seed
        )
    else:
        # Original behavior across all clients
        n_clients = len(sats)
        parts = iid_partition(trainset, n_clients) if args.iid else noniid_partition_by_label(trainset, n_clients, 2)


    def hist_for_indices(ds, idxs, max_show=10):
        ys = [int(ds[i][1]) for i in idxs]
        cnt = Counter(ys)
        uniq = sorted(cnt.keys())
        print(f"[sanity] shard size={len(idxs)}, unique labels={uniq}")
        print(f"[sanity] per-class counts:", {k:int(cnt[k]) for k in uniq})
    
    # Example: print for the very first uploader (sid=0) and a couple more
    for s in sats:
        if s.sid in (9,8,5):  # tweak which sids you want to inspect
            print(f"[sanity] Satellite sid={s.sid}, shell={getattr(s,'shell',None)}")
            hist_for_indices(trainset, parts[s.sid])

    client_loaders = make_client_loaders(trainset, parts, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    test_loader = make_test_loader(testset, batch_size=128, num_workers=args.workers)
    



    last_pull_time = {s.sid: 0.0 for s in sats}
    last_pulled_theta = {s.sid: {k: v.detach().clone() for k, v in theta_global.items()} for s in sats}
    last_contact_end = {s.sid: None for s in sats}   # for budgeting on next cycle
    n_total = sum(s.local_data_size for s in sats)

    # Per-satellite runtime metadata
    last_contact_start = {s.sid: None for s in sats}
    last_upload_time = {s.sid: None for s in sats}
    last_steps = {s.sid: 0 for s in sats}

    # Keep accuracy history for plotting
    eval_history = []  # list of (t_minutes, applied_updates, acc_float_0to1, reason)
    # Track how many times theta_global is actually modified ("applied updates")
    applied_updates = 0

    # Event lines that will be printed alongside the next evaluation line
    events_since_eval = []  # strings

    def do_eval(t_seconds, reason:str):
        # Evaluate and flush pending event logs alongside this eval
        mtmp = build_model(args.dataset, args.model).to(device)
        mtmp.load_state_dict(theta_global)
        acc = evaluate(mtmp, test_loader, device)
        t_min = t_seconds/60.0
        eval_history.append((t_min, applied_updates, acc, reason))
        print(f"[t={t_min:.1f} min | updates={applied_updates}] acc={acc*100:.2f}% algo={args.algo} reason={reason}")
        if events_since_eval:
            print("—— recent events ——")
            for line in events_since_eval:
                print(line)
            print("——————")
            events_since_eval.clear()

    next_eval_t = args.eval_every_min * 60.0

    # Per-satellite models
    sat_models = {}
    for s in sats:
        sm = build_model(args.dataset, args.model).to(device)
        sm.load_state_dict(theta_global)
        sm.will_train = False
        sm.local_buffer = None
        sat_models[s.sid] = sm

    # Map loaders to sids in sats order
    sat_loader = {sid: client_loaders[i] for i, sid in enumerate([s.sid for s in sats])}
    pending_updates = {}  # sid -> (state_dict, weight)

    def try_sync_aggregate(current_t=None):
        nonlocal applied_updates
        nonlocal theta_global, pending_updates, last_pulled_theta
        if len(pending_updates) == len(sats):
            local_thetas, weights = zip(*pending_updates.values())
            fedavg_aggregate(local_thetas, weights, theta_global)
            applied_updates += 1
            # After a synchronous aggregate, evaluate if desired
            if args.eval_mode in ("updates","contact_end"):
                if args.eval_mode == "contact_end" or (applied_updates % args.eval_stride_updates == 0):
                    t_eval = current_t if current_t is not None else (next_eval_t)
                    do_eval(t_eval, reason=("aggregate" if args.eval_mode=="contact_end" else "updates"))
            # refresh pulled thetas
            for s in sats:
                last_pulled_theta[s.sid] = {k: v.detach().clone() for k, v in theta_global.items()}
            pending_updates.clear()

    loop = EventLoop(sats, horizon_min=args.horizon_min)
    print(f"[dbg] queued events: {len(loop.pq)}")

    def budgeted_steps(sid:int, now_t:float, mode:str):
        s = sid2sat[sid]
        if args.sec_per_step <= 0:
            return args.local_steps
        if mode == "between":
            if last_contact_end[sid] is not None:
                budget_sec = max(0.0, now_t - last_contact_end[sid])
            else:
                budget_sec = max(0.0, (s.period_min - s.contact_duration_min) * 60.0)
        else:
            budget_sec = max(0.0, s.contact_duration_min * 60.0)
        max_steps = int(budget_sec // args.sec_per_step)
        return max(0, min(args.local_steps, max_steps))

    def log_contact_start(t, sid, k_pass):
        s = sid2sat[sid]
        last_contact_start[sid] = t
        msg = (f"[fed] t={t/60.0:.1f} min sid={sid} contact_start "
               f"(pass={k_pass}, period={s.period_min:.1f} min, window={s.contact_duration_min:.1f} min)")
        print(msg)
        events_since_eval.append(msg)

    def log_upload(t, sid, steps_used, algo, weight=None, delta_t=None):
        last_upload_time[sid] = t
        bits = [f"[fed] t={t/60.0:.1f} min sid={sid} uploaded update", f"algo={algo}", f"steps={steps_used}"]
        if weight is not None:
            bits.append(f"weight={weight}")
        if delta_t is not None:
            bits.append(f"Δt={delta_t/60.0:.2f} min")
        msg = " ".join(bits)
        print(msg)
        events_since_eval.append(msg)

    def on_event(t, etype, sid, k_pass):
        nonlocal theta_global, next_eval_t, applied_updates
        s = sid2sat[sid]
        
        if args.train_between_passes:
            if etype == "contact_start":
                ok = _pass_success(sid, k_pass)
            
                # (A) Do the "between-pass" local training now, starting from the *current* local model.
                if getattr(sat_models[sid], "will_train", False):
                    steps = budgeted_steps(sid, t, mode="between")
                    if steps > 0:
                        theta_local = satellite_local_train(
                            sat_models[sid],                    # train IN-PLACE from latest local θ
                            last_pulled_theta[sid],            # prox anchor (ignored if lam==0)
                            sat_loader[sid],
                            steps=steps, lr=args.lr, lam=args.lam, device=device
                        )
                        # Persist local progress and stage payload for upload
                        sat_models[sid].load_state_dict(theta_local)
                        sat_models[sid].local_buffer = theta_local
                        last_steps[sid] = steps
                    # we just consumed the "between" budget for this gap
                    sat_models[sid].will_train = False
            
                # (B) If the contact FAILS, do not pull global, do not log. Keep improved local θ.
                if not ok:
                    # arm for the *next* gap (the sat will keep training offline until next pass)
                    sat_models[sid].will_train = True
                    return
            
                # (C) Successful contact: pull the current global and arm next gap
                sat_models[sid].load_state_dict(theta_global)
                last_pull_time[sid] = t
                last_pulled_theta[sid] = {k: v.detach().clone() for k, v in theta_global.items()}
                sat_models[sid].will_train = True
                log_contact_start(t, sid, k_pass)


            elif etype == "contact_end":
                ok = _pass_success(sid, k_pass)
                if not ok:
                    return
                else:
                    if sat_models[sid].local_buffer is not None:
                        theta_local = sat_models[sid].local_buffer
                        if args.algo == "fedavg_sync":
                            pending_updates[sid] = (theta_local, s.local_data_size)
                            log_upload(t, sid, last_steps[sid], args.algo, weight=s.local_data_size)
                            try_sync_aggregate(current_t=t)
                        elif args.algo == "fedasync":
                            delta_t = t - last_pull_time[sid]
                            log_upload(t, sid, last_steps[sid], args.algo, delta_t=delta_t)
                            _alpha, _st = fedasync_update(theta_global, theta_local, delta_t,
                                          args.alpha_base, args.eps, args.To_max_min, args.a)
                            applied_updates += 1
                            if args.eval_mode in ("updates","contact_end"):
                                if args.eval_mode == "contact_end" or (applied_updates % args.eval_stride_updates == 0):
                                    do_eval(t, reason=("contact_end" if args.eval_mode=="contact_end" else "updates"))
                        elif args.algo == "fedsat":
                            log_upload(t, sid, last_steps[sid], args.algo, weight=s.local_data_size)
                            fedsat_update(theta_global, last_pulled_theta[sid], theta_local, nk=s.local_data_size, n_total=n_total)
                            applied_updates += 1
                            if args.eval_mode in ("updates","contact_end"):
                                if args.eval_mode == "contact_end" or (applied_updates % args.eval_stride_updates == 0):
                                    do_eval(t, reason=("contact_end" if args.eval_mode=="contact_end" else "updates"))
                    last_contact_end[sid] = t

        else:
            if etype == "contact_start":
                sat_models[sid].load_state_dict(theta_global)
                last_pull_time[sid] = t
                steps = budgeted_steps(sid, t, mode="contact")
                if steps > 0:
                    theta_local = satellite_local_train(
                        sat_models[sid], theta_global, sat_loader[sid],
                        steps=steps, lr=args.lr, lam=args.lam, device=device
                    )
                else:
                    theta_local = {k: v.detach().clone() for k, v in sat_models[sid].state_dict().items()}
                sat_models[sid].local_buffer = theta_local
                last_steps[sid] = steps
                log_contact_start(t, sid, k_pass)

            elif etype == "contact_end":
                theta_local = sat_models[sid].local_buffer
                if args.algo == "fedavg_sync":
                    pending_updates[sid] = (theta_local, s.local_data_size)
                    log_upload(t, sid, last_steps[sid], args.algo, weight=s.local_data_size)
                    try_sync_aggregate(current_t=t)
                elif args.algo == "fedasync":
                    delta_t = t - last_pull_time[sid]
                    log_upload(t, sid, last_steps[sid], args.algo, delta_t=delta_t)
                    _alpha, _st = fedasync_update(theta_global, theta_local, delta_t,
                                  args.alpha_base, args.eps, args.To_max_min, args.a)
                    applied_updates += 1
                    if args.eval_mode in ("updates","contact_end"):
                        if args.eval_mode == "contact_end" or (applied_updates % args.eval_stride_updates == 0):
                            do_eval(t, reason=("contact_end" if args.eval_mode=="contact_end" else "updates"))
                elif args.algo == "fedsat":
                    log_upload(t, sid, last_steps[sid], args.algo, weight=s.local_data_size)
                    fedsat_update(theta_global, last_pulled_theta[sid], theta_local, nk=s.local_data_size, n_total=n_total)
                    applied_updates += 1
                    if args.eval_mode in ("updates","contact_end"):
                        if args.eval_mode == "contact_end" or (applied_updates % args.eval_stride_updates == 0):
                            do_eval(t, reason=("contact_end" if args.eval_mode=="contact_end" else "updates"))

        if args.eval_mode == "time" and t >= next_eval_t:
            do_eval(t, reason="time")
            next_eval_t += args.eval_every_min * 60.0

        if args.algo == "fedavg_sync" and len(pending_updates) >= len(sats):
            local_thetas, weights = zip(*pending_updates.values())
            fedavg_aggregate(local_thetas, weights, theta_global)
            pending_updates.clear()

    do_eval(0.0, reason="init")
    _pass_ok = {}
    
    def _pass_success(sid: int, k: int) -> bool:
        """
        Deterministic Bernoulli for pass (sid, k) using args.seed.
        We use a local RNG seeded by a simple hash of (seed, sid, k).
        """
        key = (sid, k)
        if key in _pass_ok:
            return _pass_ok[key]
        # deterministic seed per pass:
        rnd = random.Random((args.seed + 13) * (sid + 1) * (k + 7))
        ok = (rnd.random() < args.p_comm_success)
        _pass_ok[key] = ok
        return ok
    
    loop.run(on_event)

    mtmp = build_model(args.dataset, args.model).to(device)
    mtmp.load_state_dict(theta_global)
    final_acc = evaluate(mtmp, test_loader, device)
    print(f"[final t={args.horizon_min:.1f} min] test acc={final_acc*100:.2f}% algo={args.algo}")
    eval_history.append((args.horizon_min, applied_updates, final_acc, "final"))

    if eval_history:
        if args.eval_mode == "updates":
            xs = [u for (_t, u, _a, _r) in eval_history]
            x_label = "Applied global updates"
        else:
            xs = [t for (t, _u, _a, _r) in eval_history]
            x_label = "Simulated time (minutes)"
        ys = [100.0*a for (_t, _u, a, _r) in eval_history]
        rsn = [r for (_t, _u, _a, r) in eval_history]

        csv_path = os.path.join(args.out_dir, f"{run_slug}__{timestamp}.csv")
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write(f"# {run_slug}\n")
            f.write("x,accuracy,reason\n")
            for x, y, r in zip(xs, ys, rsn):
                f.write(f"{x:.6f},{y:.6f},{r}\n")
        print(f"[save] wrote {csv_path}")
        
        plt.figure(figsize=(7.5, 4.5))
        plt.plot(xs, ys, marker='o')
        plt.xlabel(x_label)
        plt.ylabel("Test accuracy (%)")
        title_bits = [f"algo={args.algo}", f"preset={args.preset}", f"model={args.model}", f"dataset={args.dataset}"]
        if args.run_name:
            title_bits.append(f"run={args.run_name}")
        plt.title("Accuracy trajectory — " + ", ".join(title_bits))
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.savefig("accuracy_over_time.png", dpi=150)
        try:
            plt.show()
        except Exception:
            pass
    else:
        print("[plot] No evaluation points collected; nothing to plot.")


if __name__ == "__main__":
    main()

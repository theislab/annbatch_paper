from __future__ import annotations

import argparse
import contextlib
import time
from pathlib import Path

import numpy as np

import _strategies
from _common import TASKS, LabelTable, configure_zarr, load_config, provenance, write_result

def prefetch(iterable, depth: int = 4):
    """Run the loader on a background thread so reads overlap with the GPU step."""
    import queue
    import threading

    q: queue.Queue = queue.Queue(maxsize=depth)
    sentinel = object()

    def producer():
        try:
            for item in iterable:
                q.put(item)
        except Exception as exc:  # noqa: BLE001 - re-raised on the consumer side
            q.put(exc)
        finally:
            q.put(sentinel)

    t = threading.Thread(target=producer, daemon=True)
    t.start()
    while True:
        item = q.get()
        if item is sentinel:
            return
        if isinstance(item, Exception):
            raise item
        yield item


SCDATASET_REFERENCE_LR = 1e-5
SCDATASET_REFERENCE_BATCH = 64
TARGET_SUM = 1e4  # counts-per-10k, then log1p -- the scanpy default


# model
def build_models(n_genes: int, n_classes: dict[str, int], device):
    import torch
    from torch import nn

    linear = nn.ModuleDict({t: nn.Linear(n_genes, c) for t, c in n_classes.items()})
    mlp = nn.ModuleDict(
        {
            t: nn.Sequential(
                nn.Linear(n_genes, 512), nn.GELU(),
                nn.Linear(512, 512), nn.GELU(),
                nn.Linear(512, c),
            )
            for t, c in n_classes.items()
        }
    )
    model = nn.ModuleDict({"linear": linear, "mlp": mlp}).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] {n_params / 1e6:.1f}M parameters across "
          f"{2 * len(n_classes)} independent models", flush=True)
    return model


def normalise(x_dense, target_sum: float = TARGET_SUM):
    """log1p(counts per `target_sum`) -- the standard scanpy normalisation."""
    import torch

    totals = x_dense.sum(dim=1, keepdim=True).clamp(min=1.0)
    return torch.log1p(x_dense * (target_sum / totals))


# evaluation
def _macro_f1_from_confusion(cm):
    """Macro-F1 over the classes that occur in the reference labels."""
    tp = cm.diagonal().astype(np.float64)
    support = cm.sum(axis=1).astype(np.float64)
    predicted = cm.sum(axis=0).astype(np.float64)
    present = support > 0
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)
    return float(f1[present].mean()), int(present.sum())


@contextlib.contextmanager
def _eval_mode(model):
    was = model.training
    model.eval()
    try:
        yield
    finally:
        model.train(was)


def evaluate(model, X, y, n_classes, device, *, chunk: int = 4096):
    """Loss, accuracy and macro-F1 for every (architecture, task) pair."""
    import torch
    from torch.nn import functional as F  # noqa: N812

    cms = {(a, t): np.zeros((n_classes[t], n_classes[t]), dtype=np.int64)
           for a in ("linear", "mlp") for t in TASKS}
    loss_sum = {k: 0.0 for k in cms}
    n_total = X.shape[0]

    with torch.no_grad(), _eval_mode(model):
        for s in range(0, n_total, chunk):
            xb = torch.as_tensor(X[s : s + chunk].toarray(), device=device)
            xb = normalise(xb)
            for arch in ("linear", "mlp"):
                for t in TASKS:
                    yb = torch.as_tensor(y[t][s : s + chunk].astype(np.int64), device=device)
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                        logits = model[arch][t](xb)
                    logits = logits.float()
                    loss_sum[arch, t] += F.cross_entropy(logits, yb, reduction="sum").item()
                    pred = logits.argmax(1)
                    c = n_classes[t]
                    flat = (yb * c + pred).cpu().numpy()
                    cms[arch, t] += np.bincount(flat, minlength=c * c).reshape(c, c)

    out = {}
    for (arch, t), cm in cms.items():
        f1, n_present = _macro_f1_from_confusion(cm)
        out[f"{arch}/{t}"] = {
            "loss": loss_sum[arch, t] / n_total,
            "accuracy": float(cm.diagonal().sum() / cm.sum()),
            "macro_f1": f1,
            "n_classes_present": n_present,
        }
    return out


# --------------------------------------------------------------------------- #
def main() -> None:  # noqa: PLR0915 - a training script is linear by nature
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--strategy", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--lr", type=float, default=None,
                   help="default: 1e-5 * batch_size / 64 (linear scaling from scDataset's setting)")
    p.add_argument("--max-steps", type=int, default=-1, help="-1 = one full epoch")
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--num-workers", type=int, default=4, help="scDataset arms only")
    p.add_argument("--prefetch", type=int, default=4, help="batches held in the prefetch queue")
    p.add_argument("--buffer-rows", type=int, default=_strategies.DEFAULT_BUFFER_ROWS,
                   help="annbatch arms: chunk_size x preload_nchunks, held constant across the sweep")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    import torch
    import scipy.sparse as sp
    from torch.nn import functional as F  # noqa: N812

    cfg = load_config()
    configure_zarr()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lr = args.lr if args.lr is not None else SCDATASET_REFERENCE_LR * args.batch_size / SCDATASET_REFERENCE_BATCH

    labels_dir = Path(cfg["tahoe"]["labels_dir"])
    eval_dir = labels_dir.parent / "eval_sets"
    lt_train = LabelTable.load(labels_dir / "labels_train.npz")
    n_classes = lt_train.n_classes
    heldout = np.load(eval_dir / "heldout_mask_train.npy")

    val_sets = {}
    for tag in ("val_iid", "val_plate"):
        X = sp.load_npz(eval_dir / f"{tag}_X.npz")
        yz = np.load(eval_dir / f"{tag}_y.npz")
        val_sets[tag] = (X, {t: yz[t] for t in TASKS})

    run = _strategies.build(args.strategy, cfg=cfg, seed=args.seed,
                            batch_size=args.batch_size, num_workers=args.num_workers,
                            buffer_rows=args.buffer_rows)
    print(f"[train] strategy={args.strategy} store={run.store} params={run.params}", flush=True)
    print(f"[train] n_obs={run.n_obs:,d} batch_size={args.batch_size} lr={lr:g} device={device}", flush=True)

    n_genes = val_sets["val_iid"][0].shape[1]
    model = build_models(n_genes, n_classes, device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    max_steps = args.max_steps if args.max_steps > 0 else run.n_obs // args.batch_size
    print(f"[train] one epoch = {max_steps:,d} steps", flush=True)

    history: list[dict] = []
    val_history: list[dict] = []

    out = Path(args.out or Path(cfg["results"]) / "20_convergence" /
               f"{args.strategy}__bs{args.batch_size}__seed{args.seed}.json")

    def snapshot(step: int, elapsed: float, *, completed: bool) -> dict:
        """Assemble the result payload for the current step."""
        return {
            "experiment": "train_convergence",
            "strategy": args.strategy,
            "store": run.store,
            "strategy_params": run.params,
            "seed": args.seed,
            "batch_size": args.batch_size,
            "buffer_rows": args.buffer_rows,
            "lr": lr,
            "lr_rule": "1e-5 * batch_size / 64 (linear scaling from scDataset sec. 4.4)",
            "normalisation": f"log1p(counts per {TARGET_SUM:g})",
            "tasks": {t: n_classes[t] for t in TASKS},
            "n_obs_train": run.n_obs,
            # Record the GPU: per-step cost differs ~10x across node classes.
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "max_steps": max_steps,
            "steps": step,
            "completed_epoch": completed,
            "samples_seen": samples_seen,
            "elapsed_s": elapsed,
            "mean_samples_per_sec": samples_seen / elapsed if elapsed > 0 else 0.0,
            "train_history": history,
            "val_history": val_history,
            "final": val_history[-1] if val_history else None,
            "provenance": provenance(cfg),
        }
    keys = [f"{a}/{t}" for a in ("linear", "mlp") for t in TASKS]
    # Accumulate loss on the GPU; .item() per head costs eight syncs a step.
    running = torch.zeros(len(keys), device=device)
    running_n = 0
    t_start = time.perf_counter()
    t_last = t_start
    samples_seen = 0
    step = 0

    for xb_sparse, idx in prefetch(run.iterator, depth=args.prefetch):
        if step >= max_steps:
            break
        # map the pre-shuffled store's rows back into the unshuffled index space
        gidx = run.global_row[idx] if run.global_row is not None else idx
        keep = ~heldout[gidx]
        if not keep.all():
            gidx = gidx[keep]
            if gidx.size == 0:
                continue

        xb = xb_sparse.to(device, non_blocking=True).to_dense()
        if not keep.all():
            xb = xb[torch.as_tensor(np.flatnonzero(keep), device=device)]
        xb = normalise(xb)

        targets = {
            t: torch.from_numpy(lt_train.codes[t][gidx].astype(np.int64)).to(device, non_blocking=True)
            for t in TASKS
        }

        opt.zero_grad(set_to_none=True)
        total = None
        losses = []
        for arch in ("linear", "mlp"):
            for t in TASKS:
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    logits = model[arch][t](xb)
                loss = F.cross_entropy(logits.float(), targets[t])
                total = loss if total is None else total + loss
                losses.append(loss.detach())
        total.backward()
        opt.step()
        running += torch.stack(losses)

        running_n += 1
        samples_seen += int(gidx.size)
        step += 1

        if step % args.log_every == 0:
            now = time.perf_counter()
            mean_loss = dict(zip(keys, (running / running_n).tolist(), strict=True))
            history.append({
                "step": step,
                "samples": samples_seen,
                "elapsed_s": now - t_start,
                "samples_per_sec": (args.log_every * args.batch_size) / (now - t_last),
                "train_loss": mean_loss,
            })
            print(f"[train] step {step:>7,d}/{max_steps:,d}  "
                  f"lin/drug {mean_loss['linear/drug']:6.3f}  "
                  f"mlp/drug {mean_loss['mlp/drug']:6.3f}  "
                  f"{(args.log_every * args.batch_size) / (now - t_last):8.0f} samples/s", flush=True)
            running = torch.zeros(len(keys), device=device)
            running_n = 0
            t_last = now

        if step % args.eval_every == 0 or step == max_steps:
            rec = {"step": step, "samples": samples_seen, "elapsed_s": time.perf_counter() - t_start}
            for tag, (X, y) in val_sets.items():
                rec[tag] = evaluate(model, X, y, n_classes, device)
            val_history.append(rec)
            print(f"[eval ] step {step:>7,d}  "
                  f"iid mlp/drug F1={rec['val_iid']['mlp/drug']['macro_f1']:.4f}  "
                  f"plate mlp/drug F1={rec['val_plate']['mlp/drug']['macro_f1']:.4f}  "
                  f"iid mlp/cell_line F1={rec['val_iid']['mlp/cell_line']['macro_f1']:.4f}", flush=True)
            write_result(out, snapshot(step, rec["elapsed_s"],
                                       completed=step >= max_steps))
            t_last = time.perf_counter()

    elapsed = time.perf_counter() - t_start
    print(f"[train] finished {step:,d} steps in {elapsed / 3600:.3f} h", flush=True)

    write_result(out, snapshot(step, elapsed, completed=step >= max_steps))


if __name__ == "__main__":
    main()

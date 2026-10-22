#!/usr/bin/env python3
"""Train ResNet18 for NeuroSim V1.5 CIFAR-10.

Anti-disconnect:
  - Saves best state_dict to --out every improvement
  - Mirrors to --drive_dir (Google Drive) on every best + every --sync_every epochs
  - Appends epoch metrics to drive train_log.csv
  - --resume loads weights from --out or --drive_dir if present
"""
from __future__ import annotations
import argparse, csv, shutil, sys, time
from datetime import datetime, timezone
from pathlib import Path
import torch, torch.nn as nn, torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from models.resnet import resnet18


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default=str(ROOT / "datasets" / "cifar10-data"))
    p.add_argument("--out", default=str(ROOT / "models" / "resnet18_cifar10.pth"))
    p.add_argument("--drive_dir", default="",
                   help="If set, mirror best ckpt + append train_log.csv here (e.g. Drive path)")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--wd", type=float, default=5e-4)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--min_acc", type=float, default=90.0)
    p.add_argument("--sync_every", type=int, default=5,
                   help="Also copy current best to Drive every N epochs (even if not new best)")
    p.add_argument("--resume", action="store_true",
                   help="Load existing checkpoint from --out or --drive_dir before training")
    return p.parse_args()


def mirror(src: Path, drive_dir: Path, name: str):
    if not drive_dir:
        return
    drive_dir.mkdir(parents=True, exist_ok=True)
    dst = drive_dir / name
    shutil.copy2(src, dst)
    print(f"  mirrored -> {dst}")


def append_log(drive_dir: Path, row: dict):
    if not drive_dir:
        return
    drive_dir.mkdir(parents=True, exist_ok=True)
    path = drive_dir / "train_log.csv"
    write_header = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)
        f.flush()


def resolve_resume(out_path: Path, drive_dir: Path) -> Path | None:
    candidates = []
    if out_path.is_file():
        candidates.append(out_path)
    if drive_dir:
        d = drive_dir / out_path.name
        if d.is_file():
            candidates.append(d)
    return candidates[0] if candidates else None


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    drive_dir = Path(args.drive_dir) if args.drive_dir else None
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"device={device} epochs={args.epochs} out={out_path} drive_dir={drive_dir}")

    train_tf = transforms.Compose([
        transforms.Pad(4), transforms.RandomCrop(32), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    test_tf = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    Path(args.data_root).mkdir(parents=True, exist_ok=True)
    train_set = datasets.CIFAR10(args.data_root, True, download=True, transform=train_tf)
    test_set = datasets.CIFAR10(args.data_root, False, download=True, transform=test_tf)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=device.type == "cuda")
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=256, shuffle=False,
        num_workers=args.workers, pin_memory=device.type == "cuda")

    model = resnet18(num_classes=10).to(device)
    best_acc = 0.0
    if args.resume:
        ckpt = resolve_resume(out_path, drive_dir)
        if ckpt is not None:
            state = torch.load(ckpt, map_location=device)
            model.load_state_dict(state)
            print(f"resumed weights from {ckpt}")
        else:
            print("resume requested but no checkpoint found; training from scratch")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        loss_sum = correct = total = 0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)
        scheduler.step()

        model.eval()
        t_correct = t_total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                t_correct += (model(x).argmax(1) == y).sum().item()
                t_total += y.size(0)
        test_acc = 100.0 * t_correct / t_total
        train_acc = 100.0 * correct / total
        print(
            f"epoch {epoch:03d}/{args.epochs}  "
            f"loss={loss_sum/total:.4f}  train={train_acc:.2f}%  "
            f"test={test_acc:.2f}%  lr={scheduler.get_last_lr()[0]:.5f}  "
            f"{time.time()-t0:.1f}s"
        )

        improved = test_acc > best_acc
        if improved:
            best_acc = test_acc
            torch.save(model.state_dict(), out_path)
            print(f"  saved best -> {out_path} ({best_acc:.2f}%)")
            if drive_dir:
                mirror(out_path, drive_dir, out_path.name)

        if drive_dir and (improved or epoch % max(args.sync_every, 1) == 0) and out_path.is_file():
            # periodic heartbeat copy even if not improved (keeps Drive alive + latest best)
            if not improved:
                mirror(out_path, drive_dir, out_path.name)

        append_log(drive_dir or Path(), {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "epoch": epoch,
            "train_acc": f"{train_acc:.4f}",
            "test_acc": f"{test_acc:.4f}",
            "best_acc": f"{best_acc:.4f}",
            "loss": f"{loss_sum/total:.6f}",
        }) if drive_dir else None

        if test_acc >= args.min_acc:
            print(f"early stop: {test_acc:.2f}% >= {args.min_acc}")
            break

    print(f"DONE best_test_acc={best_acc:.2f}% file={out_path}")
    if drive_dir and out_path.is_file():
        mirror(out_path, drive_dir, out_path.name)


if __name__ == "__main__":
    main()

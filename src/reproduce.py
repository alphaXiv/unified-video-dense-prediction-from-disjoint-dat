#!/usr/bin/env python3
"""Compact claim-level reconstruction of UniD on public disjoint datasets.

The script deliberately prints all terminal evidence as JSON because OpenResearch
local-mode run logs are the experiment's evidence channel.
"""

from __future__ import annotations

import argparse
import copy
import io
import json
import math
import os
import random
import tarfile
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import h5py
import numpy as np
from PIL import Image
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP


NYU_URLS = [
    "https://huggingface.co/datasets/sayakpaul/nyu_depth_v2/resolve/main/data/val-000000.tar",
    "https://huggingface.co/datasets/sayakpaul/nyu_depth_v2/resolve/main/data/val-000001.tar",
]
ADE_URL = "https://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip"
RANDOMIZE_BACKBONE = False


def rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def is_main() -> bool:
    return rank() == 0


def log(event: str, **values) -> None:
    if is_main():
        print(json.dumps({"event": event, **values}, sort_keys=True), flush=True)


def download(url: str, path: Path) -> None:
    if path.exists() and path.stat().st_size > 1000:
        return
    tmp = path.with_suffix(path.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "unid-reproduction/1.0"})
    with urllib.request.urlopen(req, timeout=120) as src, tmp.open("wb") as dst:
        while True:
            chunk = src.read(8 << 20)
            if not chunk:
                break
            dst.write(chunk)
    tmp.replace(path)


def resize_rgb(array: np.ndarray, size: int) -> np.ndarray:
    return np.asarray(Image.fromarray(array.astype(np.uint8)).resize((size, size), Image.Resampling.BILINEAR))


def resize_float(array: np.ndarray, size: int) -> np.ndarray:
    t = torch.from_numpy(array.astype(np.float32))[None, None]
    return F.interpolate(t, (size, size), mode="bilinear", align_corners=False)[0, 0].numpy()


def prepare_nyu(cache: Path, cfg: dict) -> Path:
    output = cache / f"nyu_{cfg['image_size']}_{cfg['depth_train']}_{cfg['depth_eval']}.npz"
    if output.exists():
        return output
    archives = []
    for i, url in enumerate(NYU_URLS):
        target = cache / f"nyu-val-{i}.tar"
        download(url, target)
        archives.append(target)
    images, depths = [], []
    need = cfg["depth_train"] + cfg["depth_eval"]
    for archive in archives:
        with tarfile.open(archive) as tf:
            for member in tf:
                if not member.isfile() or not member.name.endswith(".h5"):
                    continue
                stream = tf.extractfile(member)
                if stream is None:
                    continue
                with h5py.File(io.BytesIO(stream.read()), "r") as h5:
                    rgb = np.asarray(h5["rgb"]).transpose(1, 2, 0)
                    depth = np.asarray(h5["depth"])
                images.append(resize_rgb(rgb, cfg["image_size"]))
                depths.append(resize_float(depth, cfg["image_size"]))
                if len(images) >= need:
                    break
        if len(images) >= need:
            break
    if len(images) < need:
        raise RuntimeError(f"NYU subset only produced {len(images)} of {need} requested examples")
    np.savez_compressed(output, images=np.stack(images), targets=np.stack(depths).astype(np.float32))
    return output


def prepare_ade(cache: Path, cfg: dict) -> Path:
    output = cache / f"ade_{cfg['image_size']}_{cfg['seg_train']}_{cfg['seg_eval']}.npz"
    if output.exists():
        return output
    archive = cache / "ADEChallengeData2016.zip"
    download(ADE_URL, archive)
    images, labels = [], []
    with zipfile.ZipFile(archive) as zf:
        for split, count in (("training", cfg["seg_train"]), ("validation", cfg["seg_eval"])):
            prefix = f"ADEChallengeData2016/images/{split}/"
            names = sorted(n for n in zf.namelist() if n.startswith(prefix) and n.endswith(".jpg"))[:count]
            for name in names:
                ann = name.replace("/images/", "/annotations/").replace(".jpg", ".png")
                with zf.open(name) as f:
                    rgb = Image.open(f).convert("RGB").resize(
                        (cfg["image_size"], cfg["image_size"]), Image.Resampling.BILINEAR
                    )
                    images.append(np.asarray(rgb))
                with zf.open(ann) as f:
                    lab = Image.open(f).resize(
                        (cfg["image_size"], cfg["image_size"]), Image.Resampling.NEAREST
                    )
                    raw = np.asarray(lab, dtype=np.int16)
                    labels.append(np.where(raw == 0, 255, raw - 1).astype(np.uint8))
    expected = cfg["seg_train"] + cfg["seg_eval"]
    if len(images) != expected:
        raise RuntimeError(f"ADE subset produced {len(images)} of {expected} requested examples")
    np.savez_compressed(output, images=np.stack(images), targets=np.stack(labels))
    return output


@dataclass
class TaskData:
    images: torch.Tensor
    targets: torch.Tensor
    train_count: int

    def batch(self, batch_size: int, generator: torch.Generator, train: bool = True):
        lo, hi = (0, self.train_count) if train else (self.train_count, len(self.images))
        idx = torch.randint(lo, hi, (batch_size,), generator=generator)
        x = self.images[idx].float().div(127.5).sub(1.0)
        y = self.targets[idx]
        return x, y


class Backbone(nn.Module):
    def __init__(self, model_id: str):
        super().__init__()
        from diffusers import UNet2DModel

        self.unet = UNet2DModel.from_pretrained(model_id, local_files_only=True)
        if RANDOMIZE_BACKBONE:
            for layer in self.unet.modules():
                reset = getattr(layer, "reset_parameters", None)
                if reset is not None:
                    reset()

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(image, (32, 32), mode="bilinear", align_corners=False)
        timesteps = torch.ones(x.shape[0], device=x.device, dtype=torch.long)
        return self.unet(x, timesteps, return_dict=False)[0]


class PixelHead(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, channels, 1),
        )

    def forward(self, latent: torch.Tensor, size: int = 64) -> torch.Tensor:
        return F.interpolate(self.net(latent), (size, size), mode="bilinear", align_corners=False)


class Specialist(nn.Module):
    def __init__(self, model_id: str, task: str):
        super().__init__()
        self.backbone = Backbone(model_id)
        self.head = PixelHead(1 if task == "depth" else 150)
        self.task = task

    def forward(self, image: torch.Tensor):
        latent = self.backbone(image)
        out = self.head(latent)
        if self.task == "depth":
            out = F.softplus(out[:, 0]) + 0.05
        return latent, out


class MultiTask(nn.Module):
    def __init__(self, model_id: str, project: bool = False):
        super().__init__()
        self.backbone = Backbone(model_id)
        self.projectors = nn.ModuleDict(
            {task: nn.Conv2d(3, 3, 1) if project else nn.Identity() for task in ("depth", "seg")}
        )
        self.heads = nn.ModuleDict({"depth": PixelHead(1), "seg": PixelHead(150)})

    def forward(self, image: torch.Tensor, task: str):
        shared = self.backbone(image)
        latent = self.projectors[task](shared)
        out = self.heads[task](latent)
        if task == "depth":
            out = F.softplus(out[:, 0]) + 0.05
        return latent, out


def depth_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    valid = target > 1e-3
    p = torch.log(pred[valid].clamp_min(1e-3))
    t = torch.log(target[valid].clamp_min(1e-3))
    d = p - t
    return torch.sqrt((d.square().mean() - 0.5 * d.mean().square()).clamp_min(1e-8))


def supervised_loss(task: str, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if task == "depth":
        return depth_loss(pred, target.float())
    return F.cross_entropy(pred, target.long(), ignore_index=255)


def cpu_state(module: nn.Module) -> dict:
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


def train_supervised(
    model: nn.Module,
    data: dict[str, TaskData],
    cfg: dict,
    steps: int,
    tasks: list[str],
    label: str,
    frozen_backbone: bool = False,
) -> dict:
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    model.to(device)
    if frozen_backbone:
        for p in model.backbone.parameters():
            p.requires_grad_(False)
    ddp = DDP(model, device_ids=[device.index], broadcast_buffers=False, find_unused_parameters=len(tasks) > 1)
    params = [p for p in ddp.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    gen = torch.Generator().manual_seed(cfg["seed"] * 1009 + rank() * 97 + 11)
    curve = []
    ddp.train()
    for step in range(steps):
        task = tasks[step % len(tasks)]
        x, y = data[task].batch(cfg["batch_per_gpu"], gen)
        x, y = x.to(device), y.to(device)
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, pred = ddp(x, task) if isinstance(model, MultiTask) else ddp(x)
            loss = supervised_loss(task, pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if step in {0, steps // 4, steps // 2, (3 * steps) // 4, steps - 1}:
            reduced = loss.detach().clone()
            dist.all_reduce(reduced)
            value = reduced.item() / dist.get_world_size()
            if is_main():
                curve.append({"step": step + 1, "loss": round(value, 6), "task": task})
    dist.barrier()
    state = cpu_state(ddp.module)
    log("training_curve", model=label, points=curve)
    del opt, ddp
    model.cpu()
    torch.cuda.empty_cache()
    return state


def train_distilled(
    teachers: dict[str, Specialist],
    cfg: dict,
    data: dict[str, TaskData],
    mode: str,
) -> dict:
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    model = MultiTask(cfg["pretrained_model"], project=True).to(device)
    for task in ("depth", "seg"):
        model.heads[task].load_state_dict(teachers[task].head.state_dict())
        for p in model.heads[task].parameters():
            p.requires_grad_(False)
        teachers[task].to(device).eval()
        for p in teachers[task].parameters():
            p.requires_grad_(False)
    ddp = DDP(model, device_ids=[device.index], broadcast_buffers=False, find_unused_parameters=True)
    params = [p for p in ddp.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    gen = torch.Generator().manual_seed(cfg["seed"] * 1009 + rank() * 97 + (31 if mode == "latent" else 53))
    curve = []
    for step in range(cfg["distill_steps"]):
        task = ("depth", "seg")[step % 2]
        x, _ = data[task].batch(cfg["batch_per_gpu"], gen)
        x = x.to(device)
        opt.zero_grad(set_to_none=True)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            teacher_latent, teacher_pixel = teachers[task](x)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            student_latent, student_pixel = ddp(x, task)
            if mode == "latent":
                loss = F.l1_loss(student_latent, teacher_latent)
            elif task == "depth":
                loss = F.l1_loss(torch.log(student_pixel), torch.log(teacher_pixel))
            else:
                teacher_prob = F.softmax(teacher_pixel.float(), dim=1)
                loss = -(teacher_prob * F.log_softmax(student_pixel.float(), dim=1)).sum(1).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if step in {0, cfg["distill_steps"] // 4, cfg["distill_steps"] // 2,
                    (3 * cfg["distill_steps"]) // 4, cfg["distill_steps"] - 1}:
            reduced = loss.detach().clone()
            dist.all_reduce(reduced)
            if is_main():
                curve.append(
                    {"step": step + 1, "loss": round(reduced.item() / dist.get_world_size(), 6), "task": task}
                )
    dist.barrier()
    state = cpu_state(ddp.module)
    log("training_curve", model=f"{mode}_distilled", points=curve)
    del opt, ddp, model
    for teacher in teachers.values():
        teacher.cpu()
    torch.cuda.empty_cache()
    return state


def finetune_projectors(state: dict, cfg: dict, data: dict[str, TaskData]) -> dict:
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    model = MultiTask(cfg["pretrained_model"], project=True)
    model.load_state_dict(state)
    for p in model.parameters():
        p.requires_grad_(False)
    for name in ("projectors", "heads"):
        for p in getattr(model, name)["seg"].parameters():
            p.requires_grad_(True)
    model.to(device)
    ddp = DDP(model, device_ids=[device.index], broadcast_buffers=False, find_unused_parameters=True)
    params = [p for p in ddp.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    gen = torch.Generator().manual_seed(cfg["seed"] * 1009 + rank() * 97 + 71)
    curve = []
    for step in range(cfg["finetune_steps"]):
        x, y = data["seg"].batch(cfg["batch_per_gpu"], gen)
        x, y = x.to(device), y.to(device)
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, pred = ddp(x, "seg")
            loss = supervised_loss("seg", pred, y)
        loss.backward()
        opt.step()
        if step in {0, cfg["finetune_steps"] // 2, cfg["finetune_steps"] - 1}:
            reduced = loss.detach().clone()
            dist.all_reduce(reduced)
            if is_main():
                curve.append({"step": step + 1, "loss": round(reduced.item() / dist.get_world_size(), 6)})
    dist.barrier()
    final = cpu_state(ddp.module)
    log("training_curve", model="latent_projector_finetune", points=curve)
    del opt, ddp, model
    torch.cuda.empty_cache()
    return final


def load_eval_model(kind: str, state: dict, cfg: dict, task: str | None = None) -> nn.Module:
    if kind == "specialist":
        model = Specialist(cfg["pretrained_model"], task or "depth")
    else:
        model = MultiTask(cfg["pretrained_model"], project=kind in {"latent", "pixel", "latent_ft"})
    model.load_state_dict(state)
    return model.cuda().eval()


@torch.no_grad()
def eval_depth(model: nn.Module, kind: str, data: TaskData, cfg: dict) -> float:
    errors = []
    batch = 16
    for start in range(data.train_count, len(data.images), batch):
        x = data.images[start : start + batch].float().div(127.5).sub(1.0).cuda()
        y = data.targets[start : start + batch].float().cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, pred = model(x, "depth") if kind != "specialist" else model(x)
        pred = pred.float()
        for p, t in zip(pred, y):
            valid = t > 1e-3
            scale = torch.median(t[valid]) / torch.median(p[valid]).clamp_min(1e-6)
            errors.append(((p[valid] * scale - t[valid]).abs() / t[valid]).mean().item())
    return float(np.mean(errors))


@torch.no_grad()
def eval_seg(model: nn.Module, kind: str, data: TaskData, cfg: dict) -> float:
    confusion = torch.zeros(150, 150, dtype=torch.int64, device="cuda")
    batch = 12
    for start in range(data.train_count, len(data.images), batch):
        x = data.images[start : start + batch].float().div(127.5).sub(1.0).cuda()
        y = data.targets[start : start + batch].long().cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, out = model(x, "seg") if kind != "specialist" else model(x)
        pred = out.argmax(1)
        valid = y != 255
        bins = torch.bincount(150 * y[valid] + pred[valid], minlength=150 * 150)
        confusion += bins.reshape(150, 150)
    diag = confusion.diag().float()
    union = confusion.sum(0) + confusion.sum(1) - diag
    present = union > 0
    return (diag[present] / union[present]).mean().item()


def boundaries(label: torch.Tensor) -> torch.Tensor:
    edge = torch.zeros_like(label, dtype=torch.bool)
    edge[:, 1:] |= label[:, 1:] != label[:, :-1]
    edge[1:, :] |= label[1:, :] != label[:-1, :]
    return edge


@torch.no_grad()
def eval_cross_task(model: nn.Module, kind: str, data: TaskData) -> float:
    scores = []
    for start in range(data.train_count, len(data.images), 8):
        x = data.images[start : start + 8].float().div(127.5).sub(1.0).cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if kind == "ensemble":
                _, depth = model["depth"](x)
                _, seg = model["seg"](x)
            else:
                _, depth = model(x, "depth")
                _, seg = model(x, "seg")
        seg = seg.argmax(1)
        gx = (depth[:, :, 1:] - depth[:, :, :-1]).abs()
        gy = (depth[:, 1:, :] - depth[:, :-1, :]).abs()
        grad = torch.zeros_like(depth)
        grad[:, :, 1:] += gx
        grad[:, 1:, :] += gy
        for d, s in zip(grad, seg):
            de = d >= torch.quantile(d.flatten(), 0.90)
            se = boundaries(s)
            se_dilated = F.max_pool2d(se.float()[None, None], 3, 1, 1)[0, 0].bool()
            de_dilated = F.max_pool2d(de.float()[None, None], 3, 1, 1)[0, 0].bool()
            precision = (de & se_dilated).sum().float() / de.sum().clamp_min(1)
            recall = (se & de_dilated).sum().float() / se.sum().clamp_min(1)
            scores.append((2 * precision * recall / (precision + recall).clamp_min(1e-8)).item())
    return float(np.mean(scores))


def evaluate_all(states: dict, cfg: dict, data: dict[str, TaskData]) -> dict:
    if not is_main():
        return {}
    results = {}
    for name, info in states.items():
        if name == "specialist":
            depth_model = load_eval_model("specialist", info["depth"], cfg, "depth")
            seg_model = load_eval_model("specialist", info["seg"], cfg, "seg")
            results[name] = {
                "depth_absrel": eval_depth(depth_model, "specialist", data["depth"], cfg),
                "seg_miou": eval_seg(seg_model, "specialist", data["seg"], cfg),
                "cross_task_edge_f1": eval_cross_task(
                    {"depth": depth_model, "seg": seg_model}, "ensemble", data["seg"]
                ),
            }
            del depth_model, seg_model
        else:
            kind = name if name in {"latent", "pixel", "latent_ft"} else "joint"
            model = load_eval_model(kind, info, cfg)
            results[name] = {
                "depth_absrel": eval_depth(model, kind, data["depth"], cfg),
                "seg_miou": eval_seg(model, kind, data["seg"], cfg),
                "cross_task_edge_f1": eval_cross_task(model, kind, data["seg"]),
            }
            del model
        torch.cuda.empty_cache()
    specialist = results["specialist"]
    for values in results.values():
        values["retention_depth"] = min(1.5, specialist["depth_absrel"] / values["depth_absrel"])
        values["retention_seg"] = values["seg_miou"] / max(specialist["seg_miou"], 1e-9)
        values["retention_mean"] = 0.5 * (values["retention_depth"] + values["retention_seg"])
    return results


class ProfileDecoder(nn.Module):
    def __init__(self, channels: int = 150):
        super().__init__()
        self.conv = nn.Conv2d(3, channels, 1)

    def forward(self, x: torch.Tensor, resolution: int):
        return F.interpolate(self.conv(x), (resolution, resolution), mode="bilinear", align_corners=False)


def profile_memory(cfg: dict) -> dict:
    if not is_main():
        return {}
    device = torch.device("cuda", 0)
    outcomes = {}
    for tasks in cfg["profile_task_counts"]:
        per_mode = {}
        for mode in ("latent", "pixel"):
            backbone = Backbone(cfg["pretrained_model"]).to(device)
            projectors = nn.ModuleList([nn.Conv2d(3, 3, 1).to(device) for _ in range(tasks)])
            decoders = nn.ModuleList([ProfileDecoder().to(device) for _ in range(tasks)])
            x = torch.randn(cfg["batch_per_gpu"], 3, 64, 64, device=device)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            baseline = torch.cuda.memory_allocated(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                shared = backbone(x)
                loss = torch.zeros((), device=device)
                for projector, decoder in zip(projectors, decoders):
                    latent = projector(shared)
                    if mode == "latent":
                        loss = loss + latent.abs().mean()
                    else:
                        pixels = decoder(latent, cfg["profile_resolution"])
                        loss = loss + pixels.square().mean()
            loss.backward()
            torch.cuda.synchronize(device)
            peak = torch.cuda.max_memory_allocated(device)
            per_mode[mode] = {
                "peak_allocated_gib": peak / (1024**3),
                "incremental_gib": (peak - baseline) / (1024**3),
            }
            if mode == "pixel":
                del pixels
            del latent, backbone, projectors, decoders, x, shared, loss
            torch.cuda.empty_cache()
        per_mode["pixel_over_latent_incremental"] = (
            per_mode["pixel"]["incremental_gib"] / max(per_mode["latent"]["incremental_gib"], 1e-9)
        )
        outcomes[str(tasks)] = per_mode
    return outcomes


def main() -> None:
    global RANDOMIZE_BACKBONE
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    RANDOMIZE_BACKBONE = bool(cfg.get("randomize_backbone", False))
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    torch.manual_seed(cfg["seed"] + rank())
    np.random.seed(cfg["seed"] + rank())
    random.seed(cfg["seed"] + rank())
    started = time.time()
    if is_main():
        log(
            "protocol",
            config=cfg,
            torch=torch.__version__,
            cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(0),
            world_size=dist.get_world_size(),
            datasets={
                "depth": "NYU Depth V2 labeled validation subset repartitioned 512 train / 142 eval",
                "segmentation": "ADE20K official training/validation subset, 800 train / 200 eval",
            },
            cutoff_qualified=True,
        )
        cache = Path("/tmp/unid-repro")
        cache.mkdir(parents=True, exist_ok=True)
        from huggingface_hub import snapshot_download

        snapshot_download(cfg["pretrained_model"])
        nyu_path = prepare_nyu(cache, cfg)
        ade_path = prepare_ade(cache, cfg)
        paths = [str(nyu_path), str(ade_path)]
    else:
        paths = [None, None]
    objects = [paths]
    dist.broadcast_object_list(objects, src=0)
    nyu_path, ade_path = objects[0]
    dist.barrier()
    nyu = np.load(nyu_path)
    ade = np.load(ade_path)
    data = {
        "depth": TaskData(torch.from_numpy(nyu["images"]).permute(0, 3, 1, 2), torch.from_numpy(nyu["targets"]), cfg["depth_train"]),
        "seg": TaskData(torch.from_numpy(ade["images"]).permute(0, 3, 1, 2), torch.from_numpy(ade["targets"]), cfg["seg_train"]),
    }
    log("data_ready", depth_examples=len(data["depth"].images), seg_examples=len(data["seg"].images))

    specialist_states = {}
    specialists = {}
    for task in ("depth", "seg"):
        model = Specialist(cfg["pretrained_model"], task)
        specialist_states[task] = train_supervised(
            model, data, cfg, cfg["specialist_steps"], [task], f"specialist_{task}"
        )
        teacher = Specialist(cfg["pretrained_model"], task)
        teacher.load_state_dict(specialist_states[task])
        specialists[task] = teacher

    frozen = MultiTask(cfg["pretrained_model"], project=False)
    frozen_state = train_supervised(
        frozen, data, cfg, cfg["joint_steps"], ["depth", "seg"], "frozen_backbone", frozen_backbone=True
    )
    joint = MultiTask(cfg["pretrained_model"], project=False)
    joint_state = train_supervised(joint, data, cfg, cfg["joint_steps"], ["depth", "seg"], "direct_joint")
    latent_state = train_distilled(specialists, cfg, data, "latent")
    pixel_state = train_distilled(specialists, cfg, data, "pixel")
    latent_ft_state = finetune_projectors(latent_state, cfg, data)

    states = {
        "specialist": specialist_states,
        "frozen": frozen_state,
        "direct_joint": joint_state,
        "latent": latent_state,
        "pixel": pixel_state,
        "latent_ft": latent_ft_state,
    }
    dist.barrier()
    results = evaluate_all(states, cfg, data)
    dist.barrier()
    memory = profile_memory(cfg)
    dist.barrier()
    if is_main():
        elapsed = (time.time() - started) / 3600
        summary = {
            "event": "FINAL_RESULT",
            "seed": cfg["seed"],
            "results": results,
            "memory_profile": memory,
            "elapsed_hours": elapsed,
            "gpu_model": torch.cuda.get_device_name(0),
            "gpu_count": dist.get_world_size(),
            "paper_reference": {
                "unid_nyu_absrel": 0.059,
                "seg_ade20k_miou_after_ft": 0.448,
                "peak_memory_latent_gib": 78,
                "peak_memory_pixel_gib_estimated": 650,
            },
        }
        print("FINAL_RESULT_JSON=" + json.dumps(summary, sort_keys=True), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

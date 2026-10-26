"""In-simulator inference for the latent MoE student (Isaac Sim's bundled torch).

Keeps each agent's previous camera frames for the 2-frame stack and blends
overlapping action chunks (temporal ensembling) into one command per step.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from lm.policy import LatentMoEPolicy  # noqa: E402


class StudentRuntime:
    def __init__(self, checkpoint: str, batch: int = 1, device: str = "cuda", ensemble_decay: float = 0.6, cuda_graph: bool = True):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.ckpt_path = str(checkpoint)
        self.task = ckpt["task"]
        cfg = ckpt["config"]
        self.cfg = cfg
        self.model = LatentMoEPolicy(**cfg).to(self.device).eval()
        self.model.load_state_dict(ckpt["state_dict"])
        self.prop_mean = torch.tensor(ckpt["norm"]["proprio_mean"], dtype=torch.float32, device=self.device)
        self.prop_std = torch.tensor(ckpt["norm"]["proprio_std"], dtype=torch.float32, device=self.device)
        self.act_scale = np.asarray(ckpt["act_scale"], dtype=np.float32)
        self.batch = batch
        self.decay = ensemble_decay
        self.reset()
        # Static input buffers; on CUDA the whole forward pass is captured as one
        # CUDA graph so a decision is a single launch instead of ~150 small kernels.
        hs, hh = ckpt.get("scene_res", 96), ckpt.get("hand_res", 64)
        self.s_scene = torch.zeros(batch, 3 * cfg["frames"], hs, hs, device=self.device)
        self.s_hand = torch.zeros(batch, 3 * cfg["frames"], hh, hh, device=self.device)
        self.s_prop = torch.zeros(batch, cfg["proprio_dim"], device=self.device)
        self.s_cmd = torch.zeros(batch, cfg["command_dim"], device=self.device) if cfg["command_dim"] else None
        self.graph = None
        with torch.no_grad():
            if self.device.type == "cuda":
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(5):
                        self._packed()
                torch.cuda.current_stream().wait_stream(stream)
                if cuda_graph:
                    self.graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(self.graph):
                        self.s_out = self._packed()
                torch.cuda.synchronize()
            else:
                for _ in range(3):
                    self._packed()

    def _packed(self):
        out = self.model(self.s_scene, self.s_hand, self.s_prop, self.s_cmd)
        n = out["act"].shape[0]
        return torch.cat([out["act"].reshape(n, -1), torch.sigmoid(out["bin_logit"]).reshape(n, -1),
                          out["phase_logit"], out["gate"], out["latent"]], dim=1).float()

    def reset(self):
        self.prev_scene = [None] * self.batch
        self.prev_hand = [None] * self.batch
        self.step = 0
        self.history = [[] for _ in range(self.batch)]   # list of (start_step, act K x A, prob K x B)

    def _stack(self, cur_list, prev_list):
        frames = []
        for i, cur in enumerate(cur_list):
            prev = prev_list[i] if prev_list[i] is not None else cur
            frames.append(np.concatenate([cur, prev], axis=-1))
        x = torch.from_numpy(np.stack(frames)).to(self.device)
        return x.permute(0, 3, 1, 2).float().div_(255.0)

    def act(self, scene_imgs, hand_imgs, proprio, command=None):
        """scene_imgs/hand_imgs: lists of HxWx3 uint8; proprio: list/array (batch x P)."""
        t0 = time.perf_counter()
        self.s_scene.copy_(self._stack(scene_imgs, self.prev_scene))
        self.s_hand.copy_(self._stack(hand_imgs, self.prev_hand))
        prop = torch.as_tensor(np.asarray(proprio, dtype=np.float32).reshape(self.batch, -1)).to(self.device)
        self.s_prop.copy_((prop - self.prop_mean) / self.prop_std)
        if self.s_cmd is not None:
            self.s_cmd.copy_(torch.as_tensor(np.asarray(command, dtype=np.float32).reshape(self.batch, -1)))
        t1 = time.perf_counter()
        with torch.no_grad():
            if self.graph is not None:
                self.graph.replay()
                packed = self.s_out.cpu().numpy()
            else:
                packed = self._packed().cpu().numpy()
        n = packed.shape[0]
        K, A, B = self.cfg["chunk"], self.cfg["act_dim"], self.cfg["bin_dim"]
        o = 0
        act = packed[:, o:o + K * A].reshape(n, K, A); o += K * A
        prob = packed[:, o:o + K * B].reshape(n, K, B); o += K * B
        phase = packed[:, o:o + self.cfg["num_phases"]].argmax(-1); o += self.cfg["num_phases"]
        gate = packed[:, o:o + self.cfg["num_experts"]]; o += self.cfg["num_experts"]
        latent = packed[:, o:]
        t2 = time.perf_counter()
        self.prev_scene = list(scene_imgs)
        self.prev_hand = list(hand_imgs)
        cmd_act = np.zeros((self.batch, act.shape[-1]), dtype=np.float32)
        cmd_bin = np.zeros((self.batch, prob.shape[-1]), dtype=np.float32)
        for i in range(self.batch):
            self.history[i].append((self.step, act[i], prob[i]))
            self.history[i] = [h for h in self.history[i] if self.step - h[0] < K]
            weights, a_sum, b_sum = 0.0, 0.0, 0.0
            for start, a, b in self.history[i]:
                age = self.step - start
                w = float(np.exp(-self.decay * age))
                a_sum = a_sum + w * a[age]
                b_sum = b_sum + w * b[age]
                weights += w
            cmd_act[i] = a_sum / weights
            cmd_bin[i] = b_sum / weights
        self.step += 1
        return {
            "act": cmd_act * self.act_scale,
            "bin": cmd_bin > 0.5,
            "bin_prob": cmd_bin,
            "phase": phase,
            "gate": gate,
            "latent": latent,
            "ms_total": (time.perf_counter() - t0) * 1000.0,
            "ms_model": (t2 - t1) * 1000.0,
        }

import subprocess
import torch
import torch.nn as nn

from datetime import datetime
from pathlib import Path

from modules.logger import create_logger

from trade.net import Net


class Model:
    def __init__(self, config, vocab, device):
        self._logger = create_logger(name="MODEL")
        self._device = device
        self._logger.info("[*] Creating model.")
        self._stats = None

        self._net = Net(config, vocab)
        self._net.to(device=self._device)

        self._optim = getattr(torch.optim, config.optim)(
            filter(lambda p: p.requires_grad, self._net.parameters()),
            **config.optim_param)

    def train(self):
        self._net.train()

    def eval(self):
        self._net.eval()

    def __call__(self, *args, **kwargs):
        if "testing" in kwargs:
            testing = kwargs.pop("testing")
        else:
            testing = True

        if testing is True:
            with torch.no_grad():
                return self._net(*args, **kwargs)
        else:
            return self._net(*args, **kwargs)

    def infer(self, *args, **kwargs):
        return self._net.infer(*args, **kwargs)

    def zero_grad(self):
        self._optim.zero_grad()

    def clip_grad(self, max_norm):
        nn.utils.clip_grad_norm_(
            filter(lambda p: p.requires_grad, self._net.parameters()),
            max_norm)

    def update(self):
        self._optim.step()

    def save_state(self, epoch, stats, ckpt_dir):
        ckpt_path = ckpt_dir / f'epoch-{epoch:0>2}.ckpt'
        torch.save({
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'epoch': epoch,
            'stats': stats,
            'net_state': self._net.state_dict(),
            'optim_state': self._optim.state_dict()}, ckpt_path)
        if self.compare(stats, self._stats):
            best_ckpt_path = ckpt_dir / 'best.ckpt'
            subprocess.call(["cp", ckpt_path, best_ckpt_path])
            self._stats = stats

    def load_state(
            self,
            ckpt_path,
            load_optim=True,
            save_device=None,
            load_device=None,
            verbose=True):
        if verbose:
            self._logger.info("[*] Load model.")
        if save_device is not None and load_device is not None:
            ckpt = torch.load(
                ckpt_path,
                map_location={f"cuda:{save_device}": f"cuda:{load_device}"})
        else:
            ckpt = torch.load(ckpt_path)
        self._net.load_state_dict(ckpt['net_state'])
        self._net.to(self._device)
        if load_optim:
            self._optim.load_state_dict(ckpt['optim_state'])
        self._stats = ckpt['stats']
        for state in self._optim.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device=self._device)
        if verbose:
            self._logger.info(f"[-] Model loaded from {ckpt_path}.")

    def load_best_state(
            self,
            ckpt_dir,
            load_optim=True,
            save_device=None,
            load_device=None):
        # ckpt_path = ckpt_dir / 'best.ckpt'
        ckpt_files = list(ckpt_dir.glob("*.ckpt"))
        ckpt_files.sort()
        ckpt_path = ckpt_files[-1]
        self.load_state(ckpt_path, load_optim, save_device, load_device)

    def compare(self, stats, best_stats):
        if not best_stats or stats['joint_acc'] > best_stats['joint_acc']:
            best_stats = stats
            return True
        else:
            return False

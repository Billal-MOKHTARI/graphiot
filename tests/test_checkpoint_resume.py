"""Checkpoints resume training exactly (task §6): stopping after k optimiser steps,
reloading the checkpoint in a fresh trainer and finishing gives bit-identical
weights, optimiser state, memory and history to an uninterrupted run — mid-epoch
and across an epoch boundary."""

import pytest
import torch

from graphiot.config.io import load_config
from graphiot.graph.dataset import build_dataset
from graphiot.model.prepare import prepare
from graphiot.model.train import Trainer

from synthetic import make_view, write_run


@pytest.fixture(scope="module")
def prep(tmp_path_factory, root):
    run = write_run(tmp_path_factory.mktemp("run"), make_view(1500.0)[1])
    cfg = load_config(root / "configs" / "dev_smoke.yaml")
    view, _ = make_view(1500.0)
    ds = build_dataset(run, cfg, root=root, view=view)
    return prep_cfg(prepare(ds, "full"), cfg)


def prep_cfg(p, cfg):
    cfg.model.batch_snapshots = 8
    cfg.model.max_epochs = 2
    cfg.model.patience = 5
    cfg.model.checkpoint_every_steps = 1
    cfg.model.d = 16
    return p, cfg


def _same(a: Trainer, b: Trainer):
    sa, sb = a.model.state_dict(), b.model.state_dict()
    assert sa.keys() == sb.keys()
    for k in sa:
        assert torch.equal(sa[k], sb[k]), k
    oa, ob = a.opt.state_dict()["state"], b.opt.state_dict()["state"]
    for k in oa:
        for kk in ("exp_avg", "exp_avg_sq"):
            assert torch.equal(oa[k][kk], ob[k][kk])
    strip = lambda hist: [{k: v for k, v in h.items() if k != "seconds"} for h in hist]
    assert strip(a.state.history) == strip(b.state.history)
    assert torch.equal(a.mem.M, b.mem.M) and a.mem.index == b.mem.index
    assert a.state.step == b.state.step and a.state.epoch == b.state.epoch


@pytest.mark.parametrize("stop_after", ["mid_epoch", "epoch_boundary"])
def test_exact_resume(prep, tmp_path, stop_after):
    p, cfg = prep
    full = Trainer(p, cfg, seed=11, out_dir=tmp_path / "full")
    full.fit()
    n_chunks = len(full.train_chunks)
    assert n_chunks >= 3 and full.state.step == 2 * n_chunks
    k = n_chunks // 2 if stop_after == "mid_epoch" else n_chunks + 1

    part = Trainer(p, cfg, seed=11, out_dir=tmp_path / "part")
    part.fit(max_steps=k)
    assert part.state.step == k and not part.state.done
    resumed = Trainer(p, cfg, seed=999, out_dir=tmp_path / "part")    # seed is restored from the checkpoint
    resumed.load(tmp_path / "part" / "last.pt")
    resumed.fit()
    _same(full, resumed)


def test_resume_refuses_other_hyperparameters(prep, tmp_path):
    p, cfg = prep
    t = Trainer(p, cfg, seed=11, out_dir=tmp_path)
    t.fit(max_steps=1)
    other = cfg.model_copy(deep=True)
    other.model.lr = 0.5
    with pytest.raises(ValueError):
        Trainer(p, other, seed=11, out_dir=tmp_path).load(tmp_path / "last.pt")

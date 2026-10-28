"""B3 worker — runs inside .venv-b3 (torch 2.2.1 + DGL 2.1.0), never imported by the
main package. It contains the E-GraphSAGE model of Lo et al. (public code
``waimorris/E-GraphSAGE``, Apache-2.0) *verbatim* — the SAGELayer / SAGE /
MLPPredictor / Model classes and the training loop are copied from the notebook
``E-GraphSAGE/netflow/bot-iot/unsw_bot_iot_binary_mean_agg.ipynb``; only the data
plumbing (read the edge tables this process is handed, write back a score per
command) is ours. See docs/DEVIATIONS.md D15: no architectural change, separate
environment because DGL cannot run next to the main venv's torch 2.14.

Usage (called by graphiot.baselines.b3_egraphsage):
    python _b3_worker.py <train.parquet> <test.parquet> <out.parquet> <meta.json>
"""

import json
import sys

import dgl
import dgl.function as fn
import networkx as nx
import numpy as np
import pandas as pd
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl import from_networkx
from sklearn.preprocessing import StandardScaler


# ----------------------------------------------------------------------------
# VERBATIM from the E-GraphSAGE notebook (Lo et al., NOMS 2022). Do not edit.
# ----------------------------------------------------------------------------
def compute_accuracy(pred, labels):
    return (pred.argmax(1) == labels).float().mean().item()


class SAGELayer(nn.Module):
    def __init__(self, ndim_in, edims, ndim_out, activation):
        super(SAGELayer, self).__init__()
        self.W_msg = nn.Linear(ndim_in + edims, ndim_out)
        self.W_apply = nn.Linear(ndim_in + ndim_out, ndim_out)
        self.activation = activation

    def message_func(self, edges):
        return {'m': self.W_msg(th.cat([edges.src['h'], edges.data['h']], 2))}

    def forward(self, g_dgl, nfeats, efeats):
        with g_dgl.local_scope():
            g = g_dgl
            g.ndata['h'] = nfeats
            g.edata['h'] = efeats
            g.update_all(self.message_func, fn.mean('m', 'h_neigh'))
            g.ndata['h'] = F.relu(self.W_apply(th.cat([g.ndata['h'], g.ndata['h_neigh']], 2)))
            return g.ndata['h']


class SAGE(nn.Module):
    def __init__(self, ndim_in, ndim_out, edim, activation, dropout):
        super(SAGE, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(SAGELayer(ndim_in, edim, 128, activation))
        self.layers.append(SAGELayer(128, edim, ndim_out, activation))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, g, nfeats, efeats):
        for i, layer in enumerate(self.layers):
            if i != 0:
                nfeats = self.dropout(nfeats)
            nfeats = layer(g, nfeats, efeats)
        return nfeats.sum(1)


class MLPPredictor(nn.Module):
    def __init__(self, in_features, out_classes):
        super().__init__()
        self.W = nn.Linear(in_features * 2, out_classes)

    def apply_edges(self, edges):
        h_u = edges.src['h']
        h_v = edges.dst['h']
        score = self.W(th.cat([h_u, h_v], 1))
        return {'score': score}

    def forward(self, graph, h):
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(self.apply_edges)
            return graph.edata['score']


class Model(nn.Module):
    def __init__(self, ndim_in, ndim_out, edim, activation, dropout):
        super().__init__()
        self.gnn = SAGE(ndim_in, ndim_out, edim, activation, dropout)
        self.pred = MLPPredictor(ndim_out, 2)

    def forward(self, g, nfeats, efeats):
        h = self.gnn(g, nfeats, efeats)
        return self.pred(g, h)


# ----------------------------------------------------------------------------
# Data plumbing (ours): edge table -> DGL graph exactly as the notebook builds it.
# ----------------------------------------------------------------------------
def build_graph(df, feat_cols, scaler, fit):
    d = df.copy()
    if fit:
        d[feat_cols] = scaler.fit_transform(d[feat_cols])
    else:
        d[feat_cols] = scaler.transform(d[feat_cols])
    d['h'] = d[feat_cols].values.tolist()
    # Notebook: nx.from_pandas_edgelist -> to_directed -> from_networkx, carrying 'h'.
    # We carry a row index too so we can recover the per-command score after prediction.
    d = d.reset_index(drop=True)
    d['eidx'] = np.arange(len(d), dtype=np.int64)
    g = nx.from_pandas_edgelist(d, "src", "dst", ['h', 'label', 'eidx'],
                                create_using=nx.MultiGraph())
    g = g.to_directed()
    g = from_networkx(g, edge_attrs=['h', 'label', 'eidx'])
    g.ndata['h'] = th.ones(g.num_nodes(), g.edata['h'].shape[1])
    g.ndata['h'] = th.reshape(g.ndata['h'], (g.ndata['h'].shape[0], 1, g.ndata['h'].shape[1]))
    g.edata['h'] = th.reshape(g.edata['h'], (g.edata['h'].shape[0], 1, g.edata['h'].shape[1])).float()
    return g


def main():
    train_p, test_p, out_p, meta_p = sys.argv[1:5]
    meta = json.load(open(meta_p))
    seed = int(meta.get("seed", 0))
    epochs = int(meta.get("epochs", 1500))
    th.manual_seed(seed)
    np.random.seed(seed)

    tr = pd.read_parquet(train_p)
    te = pd.read_parquet(test_p)
    feat_cols = [c for c in tr.columns if c.startswith("f_")]
    scaler = StandardScaler()
    G = build_graph(tr, feat_cols, scaler, fit=True)
    G_test = build_graph(te, feat_cols, scaler, fit=False)

    edge_label = G.edata['label'].long()
    # class weights (notebook: sklearn compute_class_weight 'balanced')
    classes, counts = np.unique(edge_label.numpy(), return_counts=True)
    if len(classes) == 2:
        w = len(edge_label) / (2.0 * counts)
        class_weights = th.FloatTensor([w[list(classes).index(0)] if 0 in classes else 1.0,
                                        w[list(classes).index(1)] if 1 in classes else 1.0])
    else:
        class_weights = th.FloatTensor([1.0, 1.0])
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    node_features = G.ndata['h']
    edge_features = G.edata['h']
    model = Model(G.ndata['h'].shape[2], 128, G.ndata['h'].shape[2], F.relu, 0.2)
    opt = th.optim.Adam(model.parameters())

    model.train()
    for epoch in range(1, epochs + 1):
        pred = model(G, node_features, edge_features)
        loss = criterion(pred, edge_label)
        opt.zero_grad()
        loss.backward()
        opt.step()

    model.eval()
    with th.no_grad():
        logits = model(G_test, G_test.ndata['h'], G_test.edata['h'])
        prob = F.softmax(logits, dim=1)[:, 1].numpy()           # P(attack) per directed edge
    eidx = G_test.edata['eidx'].numpy()
    # from_networkx made the graph directed (each edge both ways): average the two scores.
    score_by_eidx = pd.DataFrame({"eidx": eidx, "score": prob}).groupby("eidx")["score"].mean()
    out = te[te["command_idx"] >= 0].copy()
    out["score"] = out["eidx_scored"].map(score_by_eidx).to_numpy() if "eidx_scored" in out else np.nan
    # Map command rows to their edge row index (test edges were written in order).
    te_idx = te.reset_index(drop=True)
    te_idx["eidx"] = np.arange(len(te_idx))
    scored = te_idx[te_idx["command_idx"] >= 0]
    res = pd.DataFrame({"command_idx": scored["command_idx"].to_numpy(),
                        "score": scored["eidx"].map(score_by_eidx).to_numpy()})
    res.to_parquet(out_p, index=False)
    json.dump({"train_edges": int(len(tr)), "test_edges": int(len(te)),
               "scored": int(len(res)), "epochs": epochs, "final_train_loss": float(loss.item()),
               "train_acc": float(compute_accuracy(pred, edge_label)),
               "pos_train": int((edge_label == 1).sum())}, open(meta_p.replace(".json", "_out.json"), "w"))


if __name__ == "__main__":
    main()

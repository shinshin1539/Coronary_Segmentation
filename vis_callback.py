# vis_callback.py
from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch

# vis_callback.py
import os, csv
import numpy as np
import matplotlib.pyplot as plt

# ---------- 内部ユーティリティ ----------

def _ensure_dir(d):
    os.makedirs(d, exist_ok=True)
    return d

def _to_numpy(x):
    return x if isinstance(x, np.ndarray) else np.asarray(x)

def _squeeze_vol(v):
    """
    受け取り: (B,1,Z,Y,X) or (B,Z,Y,X) or (Z,Y,X)
    返り値:   (Z,Y,X)
    """
    v = _to_numpy(v)
    if v.ndim == 5: v = v[0, 0]
    elif v.ndim == 4: v = v[0]
    return v

def _pts_local_to_patch_xy(pts_local: np.ndarray, p: int, proj: str):
    """
    pts_local: (N,3) in [-1,1] (x,y,z)
    p: patch_size (= Z = Y = X)
    proj: 'x'|'y'|'z'   → MIP平面へ射影する2D座標を返す（imshow用）
    戻り: xs, ys
    """
    if pts_local is None or len(pts_local) == 0:
        return np.empty((0,)), np.empty((0,))
    r = p // 2
    q = pts_local * r + r              # [-1,1] → [0,p]
    q = np.clip(q, 0, p - 1e-6)
    if proj == 'x':      # MIP-X = (Z,Y)
        xs, ys = q[:, 1], q[:, 2]
    elif proj == 'y':    # MIP-Y = (Z,X)
        xs, ys = q[:, 0], q[:, 2]
    else:                # MIP-Z = (Y,X)
        xs, ys = q[:, 0], q[:, 1]
    return xs, ys

def _make_mips(vol_zyx: np.ndarray):
    """(Z,Y,X) → (MIP-X:Z×Y, MIP-Y:Z×X, MIP-Z:Y×X)"""
    return vol_zyx.max(axis=2), vol_zyx.max(axis=1), vol_zyx.max(axis=0)

# ---------- 可視化メイン ----------

def save_val_visuals(y_pred, y_true, out_dir, epoch: int, max_cases: int = 4, thr: float = 0.5):
    """
    y_pred: List[[pred_image(B,1,Z,Y,X), pred_verts(B,N,3)], ...]
    y_true: List[[true_image(B,1,Z,Y,X), true_verts(B,M,3)], ...]
    out_dir: 出力ディレクトリ（例: checkpoints/<tag>/vis）
    epoch: 現在エポック
    max_cases: 保存するケース数の上限（バッチも展開）
    thr: 予測2値化の閾値（MIP表示用）
    """
    _ensure_dir(out_dir)
    saved = 0

    for i in range(len(y_pred)):
        pred_img_b, pred_pts_b = y_pred[i]
        true_img_b, true_pts_b = y_true[i]

        pred_img_b = _to_numpy(pred_img_b)
        pred_pts_b = _to_numpy(pred_pts_b)
        true_img_b = _to_numpy(true_img_b)
        true_pts_b = _to_numpy(true_pts_b)

        B = pred_img_b.shape[0] if pred_img_b.ndim >= 4 else 1
        for b in range(B):
            # 3D パッチ
            pred_vol = _squeeze_vol(pred_img_b[b])
            true_vol = _squeeze_vol(true_img_b[b])
            p = int(true_vol.shape[0])  # 立方パッチ前提 → Z==Y==X==p

            # 2値化（見た目用）
            pred_bin = (pred_vol > thr).astype(np.float32)
            true_bin = (true_vol > 0.5).astype(np.float32)

            # MIP
            gm_x, gm_y, gm_z = _make_mips(true_bin)
            pm_x, pm_y, pm_z = _make_mips(pred_bin)

            # 点群（ローカル正規化 [-1,1] を想定）
            gt_pts_local   = true_pts_b[b] if true_pts_b.ndim == 3 else true_pts_b
            pred_pts_local = pred_pts_b[b] if pred_pts_b.ndim == 3 else pred_pts_b

            # 描画
            fig, axes = plt.subplots(2, 3, figsize=(10, 6))
            titles = ["GT MIP-X", "GT MIP-Y", "GT MIP-Z",
                      "Pred MIP-X+pts", "Pred MIP-Y+pts", "Pred MIP-Z+pts"]

            # 上段: GT MIP + GT点群（赤）
            for j, (mip, proj) in enumerate([(gm_x,'x'), (gm_y,'y'), (gm_z,'z')]):
                ax = axes[0, j]
                ax.imshow(mip, origin='lower')
                if gt_pts_local is not None and len(gt_pts_local) > 0:
                    xs, ys = _pts_local_to_patch_xy(gt_pts_local, p, proj)
                    ax.scatter(xs, ys, s=8, c='tab:red', alpha=0.9, linewidths=0.0)
                ax.set_title(titles[j]); ax.axis('off')

            # 下段: Pred MIP + Pred点群（シアン）
            for j, (mip, proj) in enumerate([(pm_x,'x'), (pm_y,'y'), (pm_z,'z')]):
                ax = axes[1, j]
                ax.imshow(mip, origin='lower')
                if pred_pts_local is not None and len(pred_pts_local) > 0:
                    xs, ys = _pts_local_to_patch_xy(pred_pts_local, p, proj)
                    ax.scatter(xs, ys, s=6, c='tab:cyan', alpha=0.8, linewidths=0.0)
                ax.set_title(titles[3+j]); ax.axis('off')

            fig.tight_layout()
            png = os.path.join(out_dir, f"epoch{int(epoch):03d}_i{i:03d}_b{b:03d}.png")
            fig.savefig(png, dpi=160); plt.close(fig)

            saved += 1
            if saved >= max_cases:
                return  # 予定数を保存したら終了

# ---------- 学習履歴の記録と曲線PNG ----------

def update_history(out_dir, epoch: int, train_loss, val_dice, val_chamfer):
    """
    checkpoints/<tag>/history.csv に追記し、curves.png を更新
    train_loss が None の場合は空欄で記録
    """
    _ensure_dir(out_dir)
    hist = os.path.join(out_dir, "history.csv")
    write_header = not os.path.exists(hist)

    with open(hist, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["epoch", "train_loss", "val_dice", "val_chamfer"])
        w.writerow([int(epoch),
                    "" if train_loss is None else float(train_loss),
                    float(val_dice), float(val_chamfer)])

    # プロット更新
    try:
        import numpy as np
        data = np.genfromtxt(hist, delimiter=",", names=True, dtype=None, encoding=None)
        if data.size == 0:
            return
        ep  = data["epoch"]
        # train_loss 列が空の場合、nan になるのでそのまま描く
        trl = data["train_loss"]
        dsc = data["val_dice"]
        chm = data["val_chamfer"]

        fig = plt.figure(figsize=(8,4.5))
        ax1 = fig.add_subplot(1,2,1); ax1.plot(ep, trl, label="Train Loss")
        ax1.set_xlabel("epoch"); ax1.set_ylabel("loss"); ax1.grid(True); ax1.legend()

        ax2 = fig.add_subplot(1,2,2); ax2.plot(ep, dsc, label="Val Dice")
        ax2.plot(ep, chm, label="Val Chamfer")
        ax2.set_xlabel("epoch"); ax2.set_ylabel("metric"); ax2.grid(True); ax2.legend()

        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "curves.png"), dpi=160)
        plt.close(fig)
    except Exception:
        # グラフ更新は失敗しても学習を止めない
        pass

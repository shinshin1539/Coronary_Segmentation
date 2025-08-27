# vis_callback.py
from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch

def _mip3(vol_zyx: np.ndarray) -> np.ndarray:
    return np.stack([vol_zyx.max(0), vol_zyx.max(1), vol_zyx.max(2)], axis=0)

def save_val_visuals(y_pred, y_true, out_dir: str, epoch: int, max_cases: int = 4, thr: float = 0.5):
    """
    y_pred: list of [pred_image(B,1,Z,Y,X), pred_verts(B,*,3)]
    y_true: list of [true_image(B,1,Z,Y,X), true_verts(B,*,3)]
    """
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    saved = 0
    for (p, t) in zip(y_pred, y_true):
        pim, pvt = p[0], p[1]
        tim, _   = t[0], t[1]
        B = pim.shape[0]
        for b in range(B):
            if saved >= max_cases: return
            pred = (pim[b,0] > thr).astype(np.uint8)  # (Z,Y,X)
            gt   = (tim[b,0] > 0.5).astype(np.uint8)

            mip_p = _mip3(pred)
            mip_g = _mip3(gt)

            P = pred.shape[0]  # cubic patch assumed
            pts = pvt[b]
            if isinstance(pts, torch.Tensor):
                pts = pts.detach().cpu().numpy()
            xyzn = (pts * (P/2.0)) + (P/2.0)
            xyzn = np.clip(xyzn, 0, P-1)
            x, y, z = xyzn[:,0], xyzn[:,1], xyzn[:,2]

            fig = plt.figure(figsize=(10,6))
            ax1 = fig.add_subplot(231); ax1.imshow(mip_g[0]); ax1.set_title('GT MIP-X'); ax1.axis('off')
            ax2 = fig.add_subplot(232); ax2.imshow(mip_g[1]); ax2.set_title('GT MIP-Y'); ax2.axis('off')
            ax3 = fig.add_subplot(233); ax3.imshow(mip_g[2]); ax3.set_title('GT MIP-Z'); ax3.axis('off')
            ax4 = fig.add_subplot(234); ax4.imshow(mip_p[0]); ax4.scatter(y, z, s=1); ax4.set_title('Pred MIP-X+pts'); ax4.axis('off')
            ax5 = fig.add_subplot(235); ax5.imshow(mip_p[1]); ax5.scatter(x, z, s=1); ax5.set_title('Pred MIP-Y+pts'); ax5.axis('off')
            ax6 = fig.add_subplot(236); ax6.imshow(mip_p[2]); ax6.scatter(x, y, s=1); ax6.set_title('Pred MIP-Z+pts'); ax6.axis('off')
            fig.tight_layout()
            fig.savefig(out / f"ep{epoch:03d}_case{saved:03d}.png", dpi=150)
            plt.close(fig)
            saved += 1

# dataset_patches.py
from __future__ import annotations
import os
from typing import List, Dict, Sequence, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset
from utils import get_json, get_medical_image, norm_zero_one

def _load_centerline_xyz(path: str, order: str = "zyx") -> np.ndarray:
    """
    Centerline .xyz を読み込み。戻りは voxel index の (x,y,z)。
    order="zyx" の場合は (z,y,x) → (x,y,z) に並べ替える。
    """
    pts = np.loadtxt(path).reshape(-1, 3)
    if order.lower() == "zyx":
        pts = pts[:, ::-1]
    return pts.astype(np.float32)  # (N,3) in (x,y,z)

def world_to_index_LPS(points_xyz_mm: np.ndarray,
                       origin_xyz: Sequence[float],
                       spacing_xyz: Sequence[float],
                       direction_flat9: Sequence[float]) -> np.ndarray:
    """
    LPS 物理座標[mm] → voxel index (x,y,z)
    SITK: phys = origin + D @ (index * spacing) なので
          index = D^T @ (phys - origin) / spacing
    """
    D = np.asarray(direction_flat9, dtype=np.float32).reshape(3,3)
    v = (points_xyz_mm.astype(np.float32) - np.asarray(origin_xyz, np.float32)).T  # (3,N)
    idx = (D.T @ v).T / np.asarray(spacing_xyz, np.float32)[None, :]
    return idx  # (N,3) (x,y,z) float

def _crop_zyx(vol_zyx: np.ndarray, center_zyx: Sequence[int], p: int) -> np.ndarray:
    """ZYXボリュームから立方体パッチ p を切り出し（必要に応じ edge でパディング）。"""
    cz, cy, cx = map(int, center_zyx)
    z0, y0, x0 = cz - p//2, cy - p//2, cx - p//2
    z1, y1, x1 = z0 + p, y0 + p, x0 + p
    Z, Y, X = vol_zyx.shape
    pad_before = (max(0, -z0), max(0, -y0), max(0, -x0))
    pad_after  = (max(0,  z1 - Z), max(0,  y1 - Y), max(0,  x1 - X))
    if any(v>0 for v in (*pad_before, *pad_after)):
        vol_zyx = np.pad(
            vol_zyx,
            ((pad_before[0], pad_after[0]),
             (pad_before[1], pad_after[1]),
             (pad_before[2], pad_after[2])),
            mode="edge"
        )
        z0 += pad_before[0]; y0 += pad_before[1]; x0 += pad_before[2]
        z1 += pad_before[0]; y1 += pad_before[1]; x1 += pad_before[2]
    return vol_zyx[z0:z1, y0:y1, x0:x1]

def _build_items(json_path: str, indexes: Sequence[int]) -> List[Dict]:
    cfg = get_json(json_path)
    root = cfg.get("dir", "")
    items: List[Dict] = []
    for k in indexes:
        for it in cfg[str(k)]:
            items.append({
                "image":      os.path.join(root, it["image"]),
                "label":      os.path.join(root, it["label"]),
                "mesh":       os.path.join(root, it["mesh"]),
                "centerline": os.path.join(root, it["centerline"]),
            })
    return items

class CenterlinePatchDataset(Dataset):
    """
    返り値:
      image: (1, p, p, p)  [0,1] float32
      label: (1, p, p, p)  {0,1} float32
      verts: (N, 3)        [-1,1] のローカル座標 (x,y,z) float32
    """
    def __init__(self,
                 items: List[Dict],
                 patch_size: int = 16,
                 patches_per_case: int = 16,
                 points_per_patch: int = 3000,
                 centerline_order: str = "zyx",            # センターラインは ZYX 入力
                 clamp_window: Tuple[float,float] = (-200, 400),
                 train: bool = True,
                 seed: int = 42):
        self.items = items
        self.p = int(patch_size)
        self.ppc = int(patches_per_case)
        self.npts = int(points_per_patch)
        self.centerline_order = centerline_order.lower()
        self.clamp = clamp_window
        self.train = train
        self.rng = np.random.default_rng(seed)

        # centerline を (x,y,z) voxel index で前読み
        self._centers_xyz: List[np.ndarray] = []
        for it in self.items:
            c = _load_centerline_xyz(it["centerline"], order=self.centerline_order)  # → (x,y,z)
            self._centers_xyz.append(c.astype(np.int32))

        # index: (case_i, center_xyz) を事前サンプリング
        self._index: List[Tuple[int, np.ndarray]] = []
        for i, cxyz in enumerate(self._centers_xyz):
            if len(cxyz) == 0:
                self._index.append((i, None))
                continue
            if self.train:
                idx = self.rng.choice(len(cxyz), size=self.ppc, replace=(len(cxyz)<self.ppc))
                picked = cxyz[idx]
            else:
                if self.ppc >= len(cxyz):
                    picked = cxyz
                else:
                    take = np.linspace(0, len(cxyz)-1, self.ppc, dtype=int)
                    picked = cxyz[take]
            for c in picked:
                self._index.append((i, c.copy()))

    def __len__(self): return len(self._index)

    def __getitem__(self, idx: int):
        case_i, c_xyz = self._index[idx]
        it = self.items[case_i]

        # 画像/ラベル（ZYX）とメタ
        img_zyx, spacing, origin, direction = get_medical_image(it["image"])
        # 画像/ラベル（ZYX）とメタ
        img_zyx, spacing, origin, direction = get_medical_image(it["image"])

        # --- SITKで読み -> >0 を 1 に二値化（スケール崩れ対策） ---
        lab_img = sitk.ReadImage(it["label"])
        lab_bin = sitk.BinaryThreshold(
            lab_img, lowerThreshold=1e-6, upperThreshold=1e20,
            insideValue=1, outsideValue=0
        )
        lab_zyx = sitk.GetArrayFromImage(lab_bin).astype(np.float32)
        img_zyx = norm_zero_one(img_zyx, span=list(self.clamp))

        # centerline が空のときは画像中心
        if c_xyz is None:
            size_xyz = np.array(img_zyx.shape[::-1], dtype=np.int32)  # (x,y,z)
            c_xyz = (size_xyz // 2).astype(np.int32)

        # パッチ切り出し（配列は ZYX）
        center_zyx = (int(c_xyz[2]), int(c_xyz[1]), int(c_xyz[0]))
        img_p = _crop_zyx(img_zyx, center_zyx, self.p)
        lab_p = _crop_zyx(lab_zyx, center_zyx, self.p)

        # GT 点群（LPS 物理座標[mm]）→ voxel index(x,y,z)
        gt_xyz_mm = np.loadtxt(it["mesh"]).reshape(-1, 3).astype(np.float32)
        gt_vox = world_to_index_LPS(gt_xyz_mm, origin, spacing, direction)

        # パッチAABBでクリップ
        r = self.p // 2
        lo = c_xyz - r
        hi = c_xyz + r
        mask = np.all((gt_vox >= lo) & (gt_vox < hi), axis=1)
        gt_local = gt_vox[mask]

        # ローカル [-1,1] へ正規化
        if len(gt_local) == 0:
            gt_local = np.zeros((1,3), np.float32)
        gt_norm = (gt_local - c_xyz) / float(r)

        # サンプル数を揃える
        sel = np.random.default_rng(int(self.rng.bit_generator.random_raw())).choice(
            len(gt_norm), size=min(self.npts, len(gt_norm)), replace=len(gt_norm)<self.npts)
        gt_norm = gt_norm[sel].astype(np.float32)

        # テンソル化（Trainer 期待形状）
        img_t = torch.from_numpy(img_p[None, ...].astype(np.float32))  # (1,p,p,p)
        lab_t = torch.from_numpy(lab_p[None, ...].astype(np.float32))  # (1,p,p,p)
        vtx_t = torch.from_numpy(gt_norm.astype(np.float32))           # (N,3)

        return img_t, lab_t, vtx_t

def get_dataset(json_path: str,
                train_indexes: Sequence[int],
                valid_indexes: Sequence[int],
                patch_size: int = 16,
                patches_per_case_train: int = 16,
                patches_per_case_val: int = 2,
                points_per_patch: int = 3000,
                centerline_order: str = "zyx",
                clamp_window: Sequence[float] = (-200, 400)):
    train_items = _build_items(json_path, train_indexes)
    valid_items = _build_items(json_path, valid_indexes)

    train_ds = CenterlinePatchDataset(
        train_items, patch_size=patch_size,
        patches_per_case=patches_per_case_train,
        points_per_patch=points_per_patch,
        centerline_order=centerline_order, clamp_window=tuple(clamp_window),
        train=True, seed=42)

    valid_ds = CenterlinePatchDataset(
        valid_items, patch_size=patch_size,
        patches_per_case=patches_per_case_val,
        points_per_patch=points_per_patch,
        centerline_order=centerline_order, clamp_window=tuple(clamp_window),
        train=False, seed=2025)

    return train_ds, valid_ds

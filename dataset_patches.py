# dataset_patches.py
from __future__ import annotations
import os
from typing import List, Dict, Sequence, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset
from utils import get_json, get_medical_image, norm_zero_one
import warnings
import nibabel as nib
from scipy.ndimage import affine_transform
import SimpleITK as sitk

import re, json

def _expand_index_spec(spec, json_path=None):
    """
    例:
      7                       → [7]
      "1-950"                → [1,2,...,950]
      "1-900,905-950"        → [1..900,905..950]
      "1-950:2"              → [1,3,5,...,949]  (step付き)
      "all"                  → data.json の全症例（'dir'キー除く）
    """
    if isinstance(spec, (list, tuple)):
        return [int(x) for x in spec]
    if spec is None:
        return []
    spec = str(spec).strip().lower()
    if spec == "all":
        if not json_path:
            return []
        with open(json_path, "r") as f:
            D = json.load(f)
        return sorted(int(k) for k in D.keys() if k != "dir")

    out = []
    for tok in re.split(r"\s*,\s*", spec):
        m = re.fullmatch(r"(\d+)\s*[-:]\s*(\d+)(?::(\d+))?", tok)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            step = int(m.group(3) or 1)
            if a <= b:
                out.extend(range(a, b + 1, step))
            else:
                out.extend(range(a, b - 1, -step))
        else:
            out.append(int(tok))
    # 重複除去して昇順
    return sorted(set(out))

def _read_label_array_nib(lbl_path: str):
    """
    nibabel でラベルを配列として読む。
    まず dataobj（無スケール）→ 全部0なら get_fdata() にフォールバック。
    返り値は XYZ 順の numpy 配列。
    """
    img = nib.load(lbl_path)
    arr = np.asanyarray(img.dataobj)
    if float(arr.max()) == 0.0:
        arr = img.get_fdata(dtype=np.float32)
    return np.asarray(arr)

def load_label_image_fixed(lbl_path: str):
    """
    1) nibabel で配列取得 (XYZ)
    2) >0.5 を 1 に二値化
    3) ZYX に転置して SimpleITK 画像化
    4) 元ラベルの spacing/origin/direction をコピー
    ※ CT と同じ幾何のはず＝サイズ一致を assert。違えば例外。
    """
    try:
        itk_lbl_orig = sitk.ReadImage(lbl_path)
    except Exception as e:
        warnings.warn(f"[Dataset] ReadImage failed: {lbl_path} ({e})")
        return None

    arr_xyz = _read_label_array_nib(lbl_path)
    if float(arr_xyz.sum()) == 0.0:
        warnings.warn(f"[Dataset] label array is all-zero: {lbl_path}")
        return None

    arr_bin = (arr_xyz > 0.5).astype(np.uint8)
    arr_zyx = np.transpose(arr_bin, (2, 1, 0)).copy()  # ← 重要：XYZ→ZYX

    itk_lbl_fix = sitk.GetImageFromArray(arr_zyx)       # (Z,Y,X) として生成
    itk_lbl_fix.CopyInformation(itk_lbl_orig)           # spacing/origin/direction を保持

    # サイズ検証（SITK の GetSize は (X,Y,Z)）
    if itk_lbl_fix.GetSize() != itk_lbl_orig.GetSize():
        raise RuntimeError(
            f"[Dataset] label size mismatch after transpose: "
            f"new={itk_lbl_fix.GetSize()} orig={itk_lbl_orig.GetSize()}"
        )
    return itk_lbl_fix

def _read_image_and_meta(path):
    """
    返り値を統一: (array_zyx, spacing_xyz, origin_xyz, direction_flat9)
    - get_medical_image が (arr, meta) でも (arr, spacing, origin, direction) でもOK
    - 最悪 get_medical_image に依らず SimpleITK でメタを取得してフォールバック
    """
    out = get_medical_image(path)
    try:
        # パターンA: (arr, spacing, origin, direction)
        if isinstance(out, tuple) and len(out) == 4:
            arr, spacing, origin, direction = out
            return arr, np.array(spacing, np.float32), np.array(origin, np.float32), np.array(direction, np.float32)
        # パターンB: (arr, meta)
        if isinstance(out, tuple) and len(out) == 2:
            arr, meta = out
            if isinstance(meta, dict):
                spacing   = meta.get("spacing")   or meta.get("Spacing")
                origin    = meta.get("origin")    or meta.get("Origin")
                direction = meta.get("direction") or meta.get("Direction")
            elif isinstance(meta, (list, tuple)) and len(meta) == 3:
                spacing, origin, direction = meta
            else:
                raise ValueError("meta must be dict or (spacing, origin, direction)")
            return arr, np.array(spacing, np.float32), np.array(origin, np.float32), np.array(direction, np.float32)
    except Exception:
        pass

    # フォールバック: SITK で読み直す
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img)
    spacing = np.array(img.GetSpacing(), np.float32)
    origin = np.array(img.GetOrigin(),  np.float32)
    direction = np.array(img.GetDirection(), np.float32)
    return arr, spacing, origin, direction

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

def _rot_scale_matrix_zyx(angle_deg: float, axis: int, scale: float = 1.0):
    """
    3D 回転＋等方スケールの 3x3 行列を ZYX 軸系で返す。
    axis: 0=Z軸まわり, 1=Y軸まわり, 2=X軸まわり
    """
    a = np.deg2rad(float(angle_deg))
    c, s = np.cos(a), np.sin(a)
    if axis == 0:
        R = np.array([[1, 0, 0],
                      [0, c, -s],
                      [0, s,  c]], dtype=np.float32)  # Z固定→Y-X面を回転
    elif axis == 1:
        R = np.array([[ c, 0, s],
                      [ 0, 1, 0],
                      [-s, 0, c]], dtype=np.float32)  # Y固定→Z-X面
    else:
        R = np.array([[ c, -s, 0],
                      [ s,  c, 0],
                      [ 0,  0, 1]], dtype=np.float32)  # X固定→Z-Y面
    return (R * float(scale)).astype(np.float32)

def _affine_on_cube_zyx(vol_zyx: np.ndarray, A_zyx: np.ndarray, order: int, mode: str = "nearest", cval: float = 0.0):
    assert vol_zyx.ndim == 3
    A = np.asarray(A_zyx, np.float32)
    A_inv = np.linalg.inv(A)
    # 体素中心を固定するためのオフセット（出力→入力の逆写像）
    c = (np.array(vol_zyx.shape, np.float32) - 1.0) / 2.0  # (cz, cy, cx)
    offset = c - A_inv @ c
    return affine_transform(vol_zyx, A_inv, offset=offset, order=order, mode=mode, cval=cval)

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
                 seed: int = 42,
                 deterministic: bool = False,
                 force_case_index: int | None = None,
                 force_center_rank: int | None = None,
                 force_center_xyz: Sequence[int] | None = None,
                 max_jitter_vox: int = 3,             # 中心ジッタ (±voxel)
                 max_rot_deg: float = 10.0,           # 回転角（絶対値上限, 度）
                 scale_range: Tuple[float,float] = (0.9, 1.1),  # 等方スケール
                 do_jitter: bool = True,
                 do_rotate: bool = True,
                 do_scale: bool  = True):
        self.items = items
        self.p = int(patch_size)
        self.ppc = int(patches_per_case)
        self.npts = int(points_per_patch)
        self.centerline_order = centerline_order.lower()
        self.clamp = clamp_window
        self.train = train
        self.rng = np.random.default_rng(seed)
        self.deterministic = deterministic   
        self.force_case_index = force_case_index
        self.force_center_rank = force_center_rank
        self.force_center_xyz  = (np.array(force_center_xyz, np.int32)
                                  if force_center_xyz is not None else None)
        self.max_jitter_vox = int(max_jitter_vox)
        self.max_rot_deg    = float(max_rot_deg)
        self.scale_range    = tuple(map(float, scale_range))
        self.do_jitter = bool(do_jitter)
        self.do_rotate = bool(do_rotate)
        self.do_scale  = bool(do_scale)

        # centerline を (x,y,z) voxel index で前読み
        self._centers_xyz: List[np.ndarray] = []
        for it in self.items:
            c = _load_centerline_xyz(it["centerline"], order=self.centerline_order)  # → (x,y,z)
            self._centers_xyz.append(c.astype(np.int32))

        # index: (case_i, center_xyz) を事前サンプリング
        self._index: List[Tuple[int, np.ndarray]] = []
        
        # ★固定パッチモード：train/val で同じ case & center を強制
        if self.force_case_index is not None:
            ci = int(self.force_case_index)
            # center を決める
            if self.force_center_xyz is not None:
                c = self.force_center_xyz
            else:
                cl = self._centers_xyz[ci]
                if len(cl) == 0:
                    # 画像中心
                    dummy = np.array(get_medical_image(self.items[ci]["image"])[0].shape[::-1]) // 2
                    c = dummy.astype(np.int32)
                else:
                    if self.force_center_rank is None:
                        # 中央の点
                        ridx = len(cl) // 2
                    else:
                        ridx = int(np.clip(self.force_center_rank, 0, len(cl)-1))
                    c = cl[ridx]
            # 同じパッチを ppc 回繰り返す（エポック内で同一）
            for _ in range(self.ppc):
                self._index.append((ci, c.copy()))
                return
            
        for i, cxyz in enumerate(self._centers_xyz):
            if len(cxyz) == 0:
                self._index.append((i, None))
                continue
            if self.train and not self.deterministic:
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
        
        # ---- 画像/メタ ----
        img_zyx, meta = get_medical_image(it["image"])  # ← utils.get_medical_image は (array, meta dict) を返す
        img_zyx = norm_zero_one(img_zyx, span=self.clamp).astype(np.float32)
        spacing   = meta['spacing']
        origin    = meta['origin']
        direction = meta['direction']
        
        # ---- ラベル（固定ローダで読み、配列に）----
        lbl_itk = load_label_image_fixed(it["label"])
        if lbl_itk is None:
            raise RuntimeError(f"label load failed or all-zero: {it['label']}")
        lab_zyx = sitk.GetArrayFromImage(lbl_itk).astype(np.float32)  # (Z,Y,X), {0,1}

        # ---- パッチ切り出し ----
        if c_xyz is None:
            size_xyz = np.array(img_zyx.shape[::-1], dtype=np.int32)  # (x,y,z)
            c_xyz = (size_xyz // 2).astype(np.int32)
            
        # ★ 中心ジッタ（学習時のみ）
        c_xyz_aug = c_xyz.copy()
        if self.train and self.do_jitter and self.max_jitter_vox > 0 and not self.deterministic:
            jitter = self.rng.integers(-self.max_jitter_vox, self.max_jitter_vox + 1, size=3)
            c_xyz_aug = (c_xyz_aug + jitter).astype(np.int32)

        center_zyx = (int(c_xyz_aug[2]), int(c_xyz_aug[1]), int(c_xyz_aug[0]))
        img_p = _crop_zyx(img_zyx, center_zyx, self.p)
        lab_p = _crop_zyx(lab_zyx, center_zyx, self.p)
        
            # ★ 回転＋スケール（学習時のみ、中心固定のアフィン）
        A_zyx = np.eye(3, dtype=np.float32)
        if self.train and not self.deterministic:
            # 回転
            if self.do_rotate and self.max_rot_deg > 0:
                ang = float(self.rng.uniform(-self.max_rot_deg, self.max_rot_deg))
                axis = int(self.rng.integers(0, 3))  # 0:Z, 1:Y, 2:X
            else:
                ang, axis = 0.0, 0
            # スケール
            if self.do_scale and (self.scale_range[0] != 1.0 or self.scale_range[1] != 1.0):
                sc = float(self.rng.uniform(self.scale_range[0], self.scale_range[1]))
            else:
                sc = 1.0

            A_zyx = _rot_scale_matrix_zyx(ang, axis, sc)
            # 画像：order=1, ラベル：order=0
            img_p = _affine_on_cube_zyx(img_p, A_zyx, order=1, mode="nearest", cval=0.0)
            lab_p = _affine_on_cube_zyx(lab_p, A_zyx, order=0, mode="constant", cval=0.0)

        # GT 点群（LPS 物理座標[mm]）→ voxel index(x,y,z)
        gt_xyz_mm = np.loadtxt(it["mesh"]).reshape(-1, 3).astype(np.float32)
        gt_vox = world_to_index_LPS(gt_xyz_mm, origin, spacing, direction)

        # パッチAABBでクリップ
        r = self.p // 2
        lo = c_xyz_aug - r
        hi = c_xyz_aug + r
        mask = np.all((gt_vox >= lo) & (gt_vox < hi), axis=1)
        gt_local = gt_vox[mask]

        # ★ 頂点にも同じ回転＋スケールを適用（中心は c_xyz_aug）
        #    注意：行列は ZYX 軸系なので、(x,y,z) → (z,y,x) に並べ替えてから変換し、終わったら戻す。
        if gt_local.size == 0:
            gt_norm = np.zeros((1, 3), np.float32)
        else:
            d_xyz = (gt_local - c_xyz_aug).astype(np.float32)       # (M,3) 相対ベクトル (x,y,z)
            d_zyx = d_xyz[:, ::-1].copy()                           # (M,3) → (z,y,x)
            d_zyx = (A_zyx @ d_zyx.T).T                             # アフィン（回転＋等方スケール）
            d_xyz_aug = d_zyx[:, ::-1].copy()                       # (x,y,z) に戻す
            gt_norm = d_xyz_aug / float(r)                          # [-1,1] 正規化

        if self.deterministic:
            if len(gt_norm) > self.npts:
                take = np.linspace(0, len(gt_norm)-1, self.npts, dtype=int)
                gt_norm = gt_norm[take]
        else:
            sel = self.rng.choice(len(gt_norm),
                                   size=min(self.npts, len(gt_norm)),
                                   replace=len(gt_norm) < self.npts)
            gt_norm = gt_norm[sel].astype(np.float32)

        # テンソル化（Trainer 期待形状）
        img_t = torch.from_numpy(img_p[None, ...].astype(np.float32))  # (1,p,p,p)
        lab_t = torch.from_numpy(lab_p[None, ...].astype(np.float32))  # (1,p,p,p)
        vtx_t = torch.from_numpy(gt_norm.astype(np.float32))           # (N,3)

        return img_t, lab_t, vtx_t

def get_dataset(*dataset_param):
    """
    trainer.py は defin_parm を dict.values() の順で *args 渡しする想定。
    与えられた個数に応じてデフォルトを補完し、順序が足りなくても動くようにする。

    期待順（左から）:
      0: json_path                        (必須)
      1: train_indexes                    (必須; 例 "1-950" や [1,2,3])
      2: valid_indexes                    (必須; 例 "951-1000" や [])
      3: patch_size                       (opt; default=16)
      4: patches_per_case_train           (opt; default=16)
      5: patches_per_case_val             (opt; default=2)
      6: points_per_patch                 (opt; default=3000)
      7: centerline_order                 (opt; default="zyx")
      8: clamp_window                     (opt; default=(-200, 400))
      9: deterministic                    (opt; default=False)
     10: force_same_patch                 (opt; default=False)
     11: force_center_rank                (opt; default=None)
     12: force_center_xyz                 (opt; default=None)
    """
    # --- helper ---
    def _get(i, default):
        return dataset_param[i] if len(dataset_param) > i and dataset_param[i] is not None else default

    def _to_bool(x, default=False):
        if isinstance(x, bool): return x
        if x is None: return default
        if isinstance(x, (int, float)): return bool(x)
        s = str(x).strip().lower()
        return s in ("1", "true", "t", "yes", "y", "on")

    # --- 必須 ---
    if len(dataset_param) < 3:
        raise ValueError(f"get_dataset expects at least 3 args, got {len(dataset_param)}")

    json_path     = _get(0, None)
    train_indexes = _get(1, None)
    valid_indexes = _get(2, None)
    if json_path is None or train_indexes is None or valid_indexes is None:
        raise ValueError("json_path, train_indexes, valid_indexes are required")

    # --- オプション ---
    patch_size             = int(_get(3, 16))
    patches_per_case_train = int(_get(4, 16))
    patches_per_case_val   = int(_get(5, 2))
    points_per_patch       = int(_get(6, 3000))
    centerline_order       =      _get(7, "zyx")
    clamp_window           = tuple(_get(8, (-200, 400)))
    deterministic          = _to_bool(_get(9,  False))
    force_same_patch       = _to_bool(_get(10, False))
    force_center_rank      = _get(11, None)
    force_center_xyz       = _get(12, None)

    # --- インデックス表記の展開（"1-950" などを配列に）---
    # ここで tuple.get(...) を使わないのがポイント！
    train_indexes = _expand_index_spec(train_indexes, json_path)
    valid_indexes = _expand_index_spec(valid_indexes, json_path)

    # --- 実データセットを構築 ---
    train_items = _build_items(json_path, train_indexes)
    valid_items = _build_items(json_path, valid_indexes)

    if force_same_patch:
        # 両方で“同じケース・同じ中心”を使って完全に同一パッチで学習/検証（過学習テスト用）
        fi_train = 0
        fi_valid = 0

        train_ds = CenterlinePatchDataset(
            train_items,
            patch_size=patch_size,
            patches_per_case=patches_per_case_train,
            points_per_patch=points_per_patch,
            centerline_order=centerline_order,
            clamp_window=clamp_window,
            train=True, seed=42,
            deterministic=True,
            force_case_index=fi_train,
            force_center_rank=force_center_rank,
            force_center_xyz=force_center_xyz
        )
        valid_ds = CenterlinePatchDataset(
            valid_items,
            patch_size=patch_size,
            patches_per_case=patches_per_case_val,
            points_per_patch=points_per_patch,
            centerline_order=centerline_order,
            clamp_window=clamp_window,
            train=False, seed=2025,
            deterministic=True,
            force_case_index=fi_valid,
            force_center_rank=force_center_rank,
            force_center_xyz=force_center_xyz
        )
        return train_ds, valid_ds

    # 通常運用
    train_ds = CenterlinePatchDataset(
        train_items,
        patch_size=patch_size,
        patches_per_case=patches_per_case_train,
        points_per_patch=points_per_patch,
        centerline_order=centerline_order,
        clamp_window=clamp_window,
        train=True, seed=42,
        deterministic=deterministic
    )
    valid_ds = CenterlinePatchDataset(
        valid_items,
        patch_size=patch_size,
        patches_per_case=patches_per_case_val,
        points_per_patch=points_per_patch,
        centerline_order=centerline_order,
        clamp_window=clamp_window,
        train=False, seed=2025,
        deterministic=deterministic
    )
    return train_ds, valid_ds
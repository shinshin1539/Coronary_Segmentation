from pathlib import Path
import numpy as np
import nibabel as nib
from skimage.measure import marching_cubes

def label_to_verts_lps(label_path: Path, level: float = 0.5, step: int = 1) -> Path:
    img = nib.load(str(label_path))
    vol = img.get_fdata().astype(np.uint8)

    # (z, y, x)
    verts_zyx, _, _, _ = marching_cubes(vol, level=level, step_size=step)

    # (x, y, z) に並べ替え → アフィン適用で RAS(mm)
    ijk = verts_zyx[:, [2, 1, 0]]
    ras = nib.affines.apply_affine(img.affine, ijk)

    # RAS → LPS
    lps = ras.copy()
    lps[:, 0] *= -1
    lps[:, 1] *= -1

    out = Path(label_path).with_name("verts.xyz")
    np.savetxt(out, lps, fmt="%.6f")
    print(f"saved: {out} (N={len(lps)})")
    return out

dir = "./data/test/"
for i in range(1, 2):
    path = dir + f"case_{i}/lab.nii.gz"
    print(path)
    label_to_verts_lps(path, level=0.5, step=1)
    print(f"Converted {path} to verts.xyz")

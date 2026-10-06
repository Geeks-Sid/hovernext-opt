"""GPU-accelerated per-tile postprocessing for HoVer-NeXt.

GPU twin of ``faster_instance_seg`` + ``post_proc_inst`` from post_process_utils,
implemented with cupy + cucim.skimage. Semantics match the CPU path:

- foreground/background thresholds per class (from best_fg/seed_thresh_cl)
- 8-connected components on seeds -> markers
- watershed on 1.0 - fg_pred restricted to the fg mask (connectivity 2)
- small-hole removal per instance + relabel of disjoint components

All functions take numpy arrays and return numpy arrays; host<->device copies are
minimal (one upload, one download per tile).
"""

import numpy as np

try:
    import cupy as cp

    from cucim.skimage import feature as csk_feature  # noqa: F401
    from cucim.skimage import measure as csk_measure
    from cucim.skimage import morphology as csk_morphology
    from cucim.skimage import segmentation as csk_segmentation

    GPU_AVAILABLE = True
except Exception:  # pragma: no cover - CPU-only hosts
    GPU_AVAILABLE = False


def _gpu():
    if not GPU_AVAILABLE:
        raise RuntimeError("cupy/cucim not available")
    return cp


def gpu_faster_instance_seg_roi(
    bg_pred, fg_pred, sem, best_fg_thresh_cl, best_seed_thresh_cl
):
    """Segment one ROI on the GPU.

    Parameters mirror the per-bbox loop body of ``faster_instance_seg``:
    bg_pred: (h, w) float array - background channel prediction
    fg_pred: (h, w) float array - foreground channel prediction
    sem:     (C, h, w) bool array - per-class argmax masks for this roi
    Returns int32 labels (h, w) with background 0, or None when the roi is degenerate.
    """
    cp = _GPU.cp

    bg = cp.asarray(bg_pred, dtype=cp.float32)
    fg = cp.asarray(fg_pred, dtype=cp.float32)
    sem_cp = cp.asarray(sem)

    ws_surface = 1.0 - fg
    fg_mask = cp.zeros(bg.shape, dtype=bool)
    seeds = cp.zeros(bg.shape, dtype=bool)

    for cl, fg_t in enumerate(best_fg_thresh_cl):
        mask = sem_cp[cl]
        fg_mask[mask] |= (1.0 - bg[mask]) > fg_t
        seeds[mask] |= fg[mask] > best_seed_thresh_cl[cl]

    n, markers = csk_measure.label(
        seeds, connectivity=2, return_num=True
    )
    if n == 0:
        return cp.zeros(bg.shape, dtype=cp.int32)

    labels = csk_segmentation.watershed(
        ws_surface, markers, mask=fg_mask, connectivity=2
    )
    return cp.ascontiguousarray(labels, dtype=cp.int32)


def gpu_post_proc_inst(pred_inst, hole_size):
    """GPU twin of ``post_proc_inst``: remove small holes per instance, then
    relabel each disconnected component. Returns (labels int32, n_final)."""
    cp = _GPU.cp

    lab = cp.asarray(pred_inst, dtype=cp.int32)
    n = int(lab.max())
    out = cp.zeros_like(lab)
    nxt = 1
    for i in range(1, n + 1):
        inst = lab == i
        filled = csk_morphology.remove_small_holes(inst, area_threshold=hole_size)
        # count the instance only if pixels survived
        if not bool(filled.any()):
            continue
        nn, relab = csk_measure.label(filled, connectivity=2, return_num=True)
        for j in range(1, nn + 1):
            out[relab == j] = nxt
            nxt += 1
    return cp.ascontiguousarray(out, dtype=cp.int32), nxt - 1


class _GPU:
    cp = None


def init_gpu():
    if GPU_AVAILABLE:
        _GPU.cp = cp
    return GPU_AVAILABLE

"""ArcFace identity embedding and embedding-guided shape refinement.

The landmark fit recovers geometry, but landmarks are a proxy: two faces can
match landmarks and still read as different people. This module closes the
loop on likeness itself: it scores the OOTP-style render of a candidate .fg
against the player's photo with a face-recognition embedding (ArcFace
w600k_r50, ONNX) and hill-climbs the FaceGen shape coefficients to maximize
that similarity (SPSA — two renders per step, no gradients needed).

The embedding model file is not bundled; see
scripts/download_restore_model.py. Note the InsightFace model-zoo weights
are distributed for non-commercial research use.
"""
from __future__ import annotations

import io
from pathlib import Path

import cv2
import numpy as np

DEFAULT_MODEL = (Path(__file__).resolve().parents[2] / "models"
                 / "arcface_w600k_r50.onnx")

# standard ArcFace 112x112 5-point template
ARC_112 = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], np.float32)

# [image-left eye, image-right eye, nose tip, left mouth, right mouth]
MP_5PT = [468, 473, 1, 61, 291]

_SESSIONS: dict[str, object] = {}


def available(model_path: Path | str | None = None) -> bool:
    return Path(model_path or DEFAULT_MODEL).is_file()


def _session(model_path: Path | str | None):
    key = str(Path(model_path or DEFAULT_MODEL))
    if key not in _SESSIONS:
        import onnxruntime as ort

        _SESSIONS[key] = ort.InferenceSession(
            key, providers=["CPUExecutionProvider"])
    return _SESSIONS[key]


def embed(img_rgb: np.ndarray, lms: np.ndarray,
          model_path: Path | str | None = None) -> np.ndarray | None:
    """L2-normalized 512-dim identity embedding, or None if alignment fails."""
    pts = lms[MP_5PT].astype(np.float32)
    M, _ = cv2.estimateAffinePartial2D(pts, ARC_112, method=cv2.LMEDS)
    if M is None:
        return None
    crop = cv2.warpAffine(img_rgb, M, (112, 112), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)
    x = (crop.astype(np.float32) - 127.5) / 127.5
    x = x.transpose(2, 0, 1)[None]
    sess = _session(model_path)
    e = sess.run(None, {sess.get_inputs()[0].name: x})[0][0]
    n = float(np.linalg.norm(e))
    return e / n if n > 1e-9 else None


def photos_embedding(images: list[np.ndarray], lms_list: list[np.ndarray],
                     model_path: Path | str | None = None,
                     weights: list[float] | None = None,
                     ) -> np.ndarray | None:
    """Weighted mean identity embedding over all usable photos."""
    if weights is not None and len(weights) != len(images):
        raise ValueError("weights length must match images")
    es = []
    ws = []
    for i, (img, lms) in enumerate(zip(images, lms_list)):
        e = embed(img, lms, model_path)
        if e is not None:
            es.append(e)
            ws.append(max(float(weights[i]) if weights is not None else 1.0, 0.0))
    if not es:
        return None
    w = np.asarray(ws, np.float32)
    if float(w.sum()) <= 1e-6:
        w = np.ones(len(es), np.float32)
    m = np.average(np.stack(es), axis=0, weights=w)
    n = float(np.linalg.norm(m))
    return m / n if n > 1e-9 else None


class _RenderScorer:
    """Renders candidate shapes against a FIXED texture/detail composite.

    Composing the OOTP texture (coefficient texture + detail-JPEG decode +
    resize) dominates render time but is constant during refinement, so it
    is built once. The ArcFace crop alignment is also frozen and only
    re-anchored every few iterations — shape deltas move landmarks by a few
    pixels at most within the refinement trust region."""

    def __init__(self, basis, tex_c: np.ndarray, detail_jpeg: bytes | None,
                 size: int, model_path):
        from .fgformat import FgFile
        from .render import _build_ootp_assets, _render_assets

        self.basis = basis
        self.size = size
        self.model_path = model_path
        self._FgFile = FgFile
        self._render_assets = _render_assets
        fg0 = FgFile(sym_shape=np.zeros(basis.n_sym),
                     asym_shape=np.zeros(basis.n_asym),
                     sym_tex=tex_c, asym_tex=np.zeros(0),
                     detail_jpeg=detail_jpeg)
        # Texture + detail are frozen across the refine, so the OOTP assets
        # (which bake them in) are built once and reused for every candidate;
        # only the shape geometry changes per render.
        self.assets = _build_ootp_assets(fg0, include_eyes=True)
        self.M = None                        # frozen 112-crop matrix

    def render(self, c: np.ndarray) -> np.ndarray:
        fg = self._FgFile(sym_shape=c[: self.basis.n_sym],
                          asym_shape=c[self.basis.n_sym :],
                          sym_tex=np.zeros(0), asym_tex=np.zeros(0),
                          detail_jpeg=None)
        return self._render_assets(self.assets, fg, self.size, shade=True, aa=1)

    def realign(self, img: np.ndarray) -> bool:
        from .landmarks import detect

        try:
            lms = detect(img)
        except Exception:
            return False
        M, _ = cv2.estimateAffinePartial2D(lms[MP_5PT].astype(np.float32),
                                           ARC_112, method=cv2.LMEDS)
        if M is None:
            return False
        self.M = M
        return True

    def embedding(self, img: np.ndarray) -> np.ndarray | None:
        if self.M is None:
            return None
        crop = cv2.warpAffine(img, self.M, (112, 112),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)
        x = (crop.astype(np.float32) - 127.5) / 127.5
        x = x.transpose(2, 0, 1)[None]
        sess = _session(self.model_path)
        e = sess.run(None, {sess.get_inputs()[0].name: x})[0][0]
        n = float(np.linalg.norm(e))
        return e / n if n > 1e-9 else None


def refine_shape(basis, photo_emb: np.ndarray, c0: np.ndarray,
                 tex_c: np.ndarray, detail_jpeg: bytes | None,
                 n_iter: int = 6, sigma: float = 0.20, lr: float = 0.75,
                 r_max: float = 2.2, reg: float = 0.0015,
                 render_size: int = 192, seed: int = 0,
                 realign_every: int = 12,
                 start_c: np.ndarray | None = None,
                 model_path: Path | str | None = None):
    """Direct identity search over the FaceGen shape modes.

    Stays within r_max (rms per-mode) of the landmark fit c0, so the
    embedding refines rather than replaces the geometric evidence. It first
    probes sparse candidate shapes, then runs a short SPSA hill climb from the
    best identity candidate. This uses only the player's photos and the local
    renderer; it does not require any external .fg identity database. start_c
    may initialize the search from a learned identity prior, while c0 remains
    the trust-region anchor and fallback.
    Returns (c_best, sim_initial, sim_best)."""
    rng = np.random.default_rng(seed)
    n_sym = basis.n_sym
    scorer = _RenderScorer(basis, tex_c, detail_jpeg, render_size, model_path)
    if not scorer.realign(scorer.render(c0)):
        return c0, float("nan"), float("nan")

    evals = [0]

    def score(c: np.ndarray) -> float:
        img = scorer.render(c)
        evals[0] += 1
        if realign_every and evals[0] % (2 * realign_every) == 0:
            scorer.realign(img)
        e = scorer.embedding(img)
        if e is None:
            return -1.0
        pen = reg * float(np.mean((c[:n_sym] - c0[:n_sym]) ** 2))
        return float(photo_emb @ e) - pen

    def clamp(c: np.ndarray) -> np.ndarray:
        c = np.clip(c, -4.0, 4.0)
        d = c[:n_sym] - c0[:n_sym]
        r = float(np.sqrt(np.mean(d ** 2)))
        if r > r_max:
            c = c.copy()
            c[:n_sym] = c0[:n_sym] + d * (r_max / r)
        return c

    def consider(candidate: np.ndarray) -> float:
        nonlocal best_c, best_f
        candidate = clamp(candidate)
        f = score(candidate)
        if f > best_f:
            best_c, best_f = candidate.copy(), f
        return f

    c_anchor = c0.astype(np.float64).copy()
    sim0 = score(c_anchor)
    best_c, best_f = c_anchor.copy(), sim0
    c = c_anchor.copy()
    if start_c is not None:
        c = clamp(np.asarray(start_c, np.float64).copy())
        consider(c)

    # Cheap beam warm-start: sparse moves are less likely to turn a face into a
    # caricature than perturbing all 50 modes at once, but they still let ArcFace
    # pull the solution toward identity cues that landmarks miss.
    n_probe = max(0, min(24, int(n_iter)))
    probe_modes = min(8, n_sym)
    for k in range(n_probe):
        d = np.zeros(n_sym, np.float64)
        idx = rng.choice(n_sym, probe_modes, replace=False)
        d[idx] = rng.normal(size=probe_modes)
        rms = float(np.sqrt(np.mean(d ** 2)))
        if rms <= 1e-9:
            continue
        d /= rms
        radius = r_max * (0.35 + 0.65 * ((k % 6) / 5.0))
        cp = c_anchor.copy()
        cp[:n_sym] += radius * d
        consider(cp)

    c = best_c.copy()
    for k in range(n_iter):
        ck = sigma / (k + 1) ** 0.101
        ak = lr / (k + 6) ** 0.602
        delta = rng.choice([-1.0, 1.0], n_sym)
        cp, cm = c.copy(), c.copy()
        cp[:n_sym] += ck * delta
        cm[:n_sym] -= ck * delta
        fp, fm = score(clamp(cp)), score(clamp(cm))
        if fp > best_f:
            best_c, best_f = clamp(cp), fp
        if fm > best_f:
            best_c, best_f = clamp(cm), fm
        g = (fp - fm) / (2.0 * ck) * delta
        c[:n_sym] += ak * g
        c = clamp(c)
    f_final = score(c)
    if f_final > best_f:
        best_c, best_f = c, f_final
    return best_c, float(sim0), float(best_f)

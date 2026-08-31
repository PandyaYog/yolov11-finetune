"""Inference wrapper for the Water-Net underwater image enhancer.

Reimplements the architecture and pre-processing from Li et al., "An
Underwater Image Enhancement Benchmark Dataset and Beyond" (IEEE TIP 2019,
https://github.com/Li-Chongyi/Water-Net_Code) against a modern TensorFlow
(tf.compat.v1) runtime. The graph is written from scratch here rather than
importing the original repo because that code is TF1/Python 2-era (matplotlib
GUI calls, tf.app.flags, MATLAB preprocessing) and won't run as-is -- but the
variable names/shapes below are copied exactly so the released checkpoint
(models/waternet_checkpoint/coarse_112/) restores into it unmodified.

The network fuses three classically-preprocessed variants of the input
(white-balanced, contrast-enhanced, gamma-corrected) through a small conv
stack that predicts a per-pixel, per-branch confidence map, then blends the
three branches by that map. It is fully convolutional (no dense layers), so
it runs at whatever resolution you feed it -- "112" in the checkpoint path is
just the training patch size, not an inference constraint.
"""

from __future__ import annotations

import glob
import importlib.util
import os
import sys
from pathlib import Path

import cv2
import numpy as np


def _wire_pip_cuda_libs() -> None:
    """`pip install tensorflow[and-cuda]` drops cuDNN/cuBLAS/etc. .so files
    inside the nvidia-* packages' own site-packages tree instead of a system
    lib dir. TF dlopen()s them by soname at first CUDA use, which only
    resolves if their directory was on LD_LIBRARY_PATH when this *process*
    started -- glibc's dynamic linker fixes its search path at exec, so
    mutating os.environ mid-run does nothing and TF silently falls back to
    CPU (~1.2s/image at 640x480 here; the RTX 4050 this was tuned against
    runs the same call in a small fraction of that). So if the libs are
    findable but not already wired, re-exec this same process with the env
    var set, guarded so it only ever happens once."""
    if os.environ.get("_UWD_CUDA_LIBS_WIRED"):
        return
    spec = importlib.util.find_spec("nvidia")
    if spec is None or not spec.submodule_search_locations:
        return
    lib_dirs = glob.glob(os.path.join(list(spec.submodule_search_locations)[0], "*", "lib"))
    if not lib_dirs:
        return
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = ":".join(lib_dirs + ([existing] if existing else []))
    os.environ["_UWD_CUDA_LIBS_WIRED"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)


_wire_pip_cuda_libs()

import tensorflow as tf  # noqa: E402

tf1 = tf.compat.v1
# The released checkpoint was written by TF1 with reference (non-resource)
# variables. TF2's default eager/resource-variable mode can't restore those,
# so drop fully into TF1 graph-mode semantics for this module.
tf1.disable_v2_behavior()


def white_balance(img_bgr: np.ndarray, percent: float = 0.5) -> np.ndarray:
    """Percentile-stretch white balance, per channel."""
    out = []
    lo_stop = img_bgr.shape[0] * img_bgr.shape[1] * percent / 200.0
    hi_stop = img_bgr.shape[0] * img_bgr.shape[1] * (1 - percent / 200.0)
    for channel in cv2.split(img_bgr):
        hist = np.cumsum(cv2.calcHist([channel], [0], None, [256], (0, 256)))
        lo, hi = np.searchsorted(hist, (lo_stop, hi_stop))
        lut = np.concatenate((
            np.zeros(lo),
            np.around(np.linspace(0, 255, max(hi - lo, 1) + 1)),
            255 * np.ones(255 - hi),
        ))[:256]
        out.append(cv2.LUT(channel, lut.astype("uint8")))
    return cv2.merge(out)


def adjust_gamma(img_bgr: np.ndarray, gamma: float = 0.7) -> np.ndarray:
    inv = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv) * 255 for i in range(256)], dtype="uint8")
    return cv2.LUT(img_bgr, table)


def contrast_enhance(img_bgr: np.ndarray) -> np.ndarray:
    """CLAHE on luminance, replicated back to 3 channels -- matches the
    original repo's ce_real preprocessing exactly."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return cv2.cvtColor(clahe.apply(gray), cv2.COLOR_GRAY2BGR)


def _conv2d(x, out_dim: int, k: int, name: str):
    with tf1.variable_scope(name):
        w = tf1.get_variable("w", [k, k, x.get_shape()[-1], out_dim],
                              initializer=tf1.truncated_normal_initializer(stddev=0.02))
        b = tf1.get_variable("biases", [out_dim], initializer=tf1.constant_initializer(0.0))
        return tf.nn.bias_add(tf1.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding="SAME"), b)


class WaterNet:
    """Loads once, enhances many same-shaped images without rebuilding the
    graph -- the original repo rebuilds per-image (its test images vary in
    size); ours are a fixed 640x480 corpus, so batching this way is a large
    speedup."""

    def __init__(self, checkpoint_dir: Path, height: int, width: int):
        self.height, self.width = height, width
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.images = tf1.placeholder(tf.float32, [1, height, width, 3], name="images")
            self.images_wb = tf1.placeholder(tf.float32, [1, height, width, 3], name="images_wb")
            self.images_ce = tf1.placeholder(tf.float32, [1, height, width, 3], name="images_ce")
            self.images_gc = tf1.placeholder(tf.float32, [1, height, width, 3], name="images_gc")
            self.output = self._model()
            saver = tf1.train.Saver()

        self.sess = tf1.Session(graph=self.graph)
        ckpt = tf1.train.get_checkpoint_state(str(checkpoint_dir))
        if not ckpt or not ckpt.model_checkpoint_path:
            raise FileNotFoundError(f"no checkpoint state found under {checkpoint_dir}")
        # get_checkpoint_state records an absolute path from wherever the
        # checkpoint was originally saved -- resolve against our copy instead
        # of trusting that path, since we almost certainly moved the folder.
        ckpt_name = Path(ckpt.model_checkpoint_path).name
        saver.restore(self.sess, str(checkpoint_dir / ckpt_name))

    def _model(self):
        with tf1.variable_scope("main_branch"):
            cat_all = tf.concat([self.images, self.images_wb, self.images_ce, self.images_gc], axis=3)
            x = tf.nn.relu(_conv2d(cat_all, 128, 7, "conv2wb_1"))
            x = tf.nn.relu(_conv2d(x, 128, 5, "conv2wb_2"))
            x = tf.nn.relu(_conv2d(x, 128, 3, "conv2wb_3"))
            x = tf.nn.relu(_conv2d(x, 64, 1, "conv2wb_4"))
            x = tf.nn.relu(_conv2d(x, 64, 7, "conv2wb_5"))
            x = tf.nn.relu(_conv2d(x, 64, 5, "conv2wb_6"))
            x = tf.nn.relu(_conv2d(x, 64, 3, "conv2wb_7"))
            confidence = tf.nn.sigmoid(_conv2d(x, 3, 3, "conv2wb_77"))

            cat_wb = tf.concat([self.images, self.images_wb], axis=3)
            y = tf.nn.relu(_conv2d(cat_wb, 32, 7, "conv2wb_9"))
            y = tf.nn.relu(_conv2d(y, 32, 5, "conv2wb_10"))
            wb_branch = tf.nn.relu(_conv2d(y, 3, 3, "conv2wb_11"))

            cat_ce = tf.concat([self.images, self.images_ce], axis=3)
            y = tf.nn.relu(_conv2d(cat_ce, 32, 7, "conv2wb_99"))
            y = tf.nn.relu(_conv2d(y, 32, 5, "conv2wb_100"))
            ce_branch = tf.nn.relu(_conv2d(y, 3, 3, "conv2wb_111"))

            cat_gc = tf.concat([self.images, self.images_gc], axis=3)
            y = tf.nn.relu(_conv2d(cat_gc, 32, 7, "conv2wb_999"))
            y = tf.nn.relu(_conv2d(y, 32, 5, "conv2wb_1000"))
            gc_branch = tf.nn.relu(_conv2d(y, 3, 3, "conv2wb_1111"))

            w_wb, w_ce, w_gc = tf.split(confidence, 3, axis=3)
            return wb_branch * w_wb + ce_branch * w_ce + gc_branch * w_gc

    def enhance(self, img_bgr: np.ndarray) -> np.ndarray:
        """img_bgr: HxWx3 uint8, BGR (cv2 convention). Returns the same."""
        assert img_bgr.shape[:2] == (self.height, self.width), \
            f"WaterNet was built for {self.height}x{self.width}, got {img_bgr.shape[:2]}"

        wb = white_balance(img_bgr, 0.5)
        ce = contrast_enhance(img_bgr)
        gc = adjust_gamma(img_bgr, 0.7)

        def feed(img):
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            return rgb[np.newaxis, ...]

        out = self.sess.run(self.output, feed_dict={
            self.images: feed(img_bgr),
            self.images_wb: feed(wb),
            self.images_ce: feed(ce),
            self.images_gc: feed(gc),
        })
        out = np.clip(out[0], 0.0, 1.0) * 255.0
        return cv2.cvtColor(out.astype(np.uint8), cv2.COLOR_RGB2BGR)

    def close(self):
        self.sess.close()

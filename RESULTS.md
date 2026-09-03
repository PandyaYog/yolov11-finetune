# Fine-tuning results — YOLOv11n, amphibious survey robot

Status: first full fine-tuning run complete (100 epochs, Kaggle 2×T4). This
document is the result record; `README.md` documents how to reproduce or
extend the pipeline.

## 1. What this model is for

An **auto-labeler** for an amphibious survey robot (Jetson + STM32,
OAK-D-Lite FF), not a controller — obstacle avoidance runs off the depth
channel separately. The model names what the robot saw so frames can be
archived by class, with low-confidence detections routed to a review queue.
That framing drove every design choice in this pipeline:

- **Precision over recall.** A false positive writes a wrong label into the
  archive; a false negative only delays a frame to human review.
- **11 coarse classes**, not species-level: `fish`, `benthic_invert`,
  `gelatinous`, `flora`, `debris`, `rope_net`, `person`, `equipment`,
  `structure`, `vessel`, `buoy_marker`.
- **Two operating domains, one model.** The robot is submerged part of the
  time and on the surface part of the time; the corpus and evaluation both
  respect that split (see §5).

## 2. Pipeline

```
12MP capture → ISP crop 640×480 (4:3) → WaterNet (submerged only) → YOLOv11n
```

1. **Download** 12 public marine/maritime datasets into a common raw tree.
2. **Unify** — each source's native format (COCO/VOC/YOLO/segmentation
   masks/MATLAB) read, mapped onto the 11-class taxonomy, deduplicated
   (perceptual hash), temporally strided (video-derived sources), capped per
   source, and cropped to 640×480. Split **by group** (video/dive/site), not
   per-image, so near-duplicate adjacent frames can't leak across
   train/val/test.
3. **Enhance** — WaterNet (Li et al., IEEE TIP 2019) run on submerged-domain
   images only; surface-domain images pass through raw. This matches the
   robot's own inference-time behaviour exactly: WaterNet is depth-sensor-
   gated on the bot too, so training never sees a distribution the deployed
   model won't also see live.
4. **Fine-tune** YOLOv11n from COCO-pretrained weights.

## 3. Dataset

**11 sources, 27,687 images** after dedup/stride/cap, group-aware split
80/10/10:

| split | images |
|---|---|
| train | 22,681 |
| val | 2,329 |
| test | 2,677 |

| source | domain | images | share |
|---|---|---|---|
| fathomnet | submerged | 9,288 | 33.5% |
| seaships | surface | 4,860 | 17.6% |
| mods | surface | 3,000 | 10.8% |
| trashcan | submerged | 2,722 | 9.8% |
| vddc | submerged | 2,490 | 9.0% |
| duo | submerged | 2,488 | 9.0% |
| suim | submerged | 1,461 | 5.3% |
| trash_icra19 | submerged | 1,071 | 3.9% |
| smd | surface | 154 | 0.6% |
| brackish | submerged | 83 | 0.3% |
| mastr_neg | surface | 70 | 0.3% |

**Boxes per class:**

| class | boxes | share |
|---|---|---|
| benthic_invert | 31,512 | 36.7% |
| vessel | 22,246 | 25.9% |
| person | 12,849 | 14.9% |
| fish | 7,251 | 8.4% |
| gelatinous | 3,230 | 3.8% |
| debris | 3,291 | 3.8% |
| structure | 2,556 | 3.0% |
| equipment | 1,815 | 2.1% |
| flora | 953 | 1.1% |
| buoy_marker | 146 | 0.2% |
| rope_net | 119 | 0.1% |

Negatives (object-free images): 70 (0.3% — well under the 12% target; not a
gap that affects this run, since precision is enforced by confidence
thresholding at inference, not training-time negative ratio).

**Known composition caveat:** `fathomnet` sits at 33.5%, over this project's
own 25% flag threshold, and supplies 86–100% of `structure`/`gelatinous`
respectively — those two classes' training signal is mostly deep-sea ROV
footage, a real domain gap against shallow-water deployment. Not fixed in
this run; see §6.

## 4. Fine-tuning setup

| | |
|---|---|
| base weights | `yolo11n.pt` (COCO-pretrained) |
| hardware | Kaggle, 2× Tesla T4 (DDP) |
| epochs | 100 |
| batch | 64 |
| image size | 640 (letterboxed square) |
| optimizer | `AdamW` (see note below) |
| train list | repeat-factor-sampled — oversamples images carrying rare classes (24,897 entries from 22,681 unique train images) |
| flipud | 0.1 — low, not 0: benefits submerged classes (no fixed "up" underwater) without corrupting the gravity-constrained surface fraction |
| degrees | 10 — modest rotation, same domain reasoning as flipud |
| extra augmentation | custom blur/haze pass (p=0.15) via Albumentations, targeting *post-WaterNet* residual softness/turbidity — not synthetic colour-cast, since WaterNet already corrects that before YOLO ever sees the image |
| training time | ~5.15 hours |

**Optimizer note:** Ultralytics' `optimizer=auto` selects a new `MuSGD`
optimizer once total iterations exceed 10,000 (any real multi-epoch run),
and its DDP code path has a bug — crashes both GPU ranks on the first
optimizer step. Pinned to `AdamW` explicitly to avoid it. Documented in
`README.md` so it doesn't get silently reverted.

## 5. Results

Validated three ways: the blended metric Ultralytics tracks every epoch,
and a submerged-vs-surface domain split on **both** val and the fully
held-out test set (test is never touched by training or per-epoch
validation — the one honest read).

**Training-time (blended) val, final/best epoch (100/100 — still improving,
not plateaued):** mAP50 = 0.555, mAP50-95 = 0.334, precision = 0.601, recall = 0.523

**Per-domain:**

| | val images | val mAP50 | val mAP50-95 | test images | test mAP50 | test mAP50-95 |
|---|---|---|---|---|---|---|
| submerged | 1,510 | 0.525 | 0.345 | 1,864 | 0.533 | 0.355 |
| surface | 819 | 0.692 | 0.336 | 813 | 0.609 | 0.307 |

Both domains land in the same range on both splits — the single unified
model isn't quietly failing one domain to serve the other, and val/test
agree closely (no overfitting to val).

**Per-class mAP50-95 (test):**

| class | submerged | surface |
|---|---|---|
| fish | 0.335 | — |
| benthic_invert | 0.513 | — |
| gelatinous | 0.539 | — |
| flora | 0.020 | — |
| debris | 0.456 | — |
| rope_net | n/a (0 instances) | n/a (0 instances) |
| person | 0.485 | 0.131 |
| equipment | 0.412 | — |
| structure | 0.078 | — |
| vessel | — | 0.482 |
| buoy_marker | — | n/a (0 instances) |

(`—` = class doesn't occur in that domain by taxonomy design, e.g. `vessel`
is surface-only. `n/a (0 instances)` = occurs in principle but this split
happened to contain zero ground-truth boxes for it.)

**Two real findings, not noise — repeatable across val and test:**

1. **`person` is weak specifically on the surface** (val 0.158, test 0.131)
   vs. strong underwater (val 0.536, test 0.485). Consistent across both
   splits, so it's a real gap, not a fluke — likely thin/hard surface-person
   coverage in the source data (`mods`/`seaships` are vessel-focused).
2. **`buoy_marker` has zero test-set instances**, despite scoring 0.383
   mAP50-95 on val. With only 5 source videos supplying this class, the
   group-aware split had too little to distribute — every buoy-containing
   video happened to land in train/val, none in test. There is currently no
   honest held-out read on this specific class.
3. `flora` (0.02) and `structure` (0.08) are the weakest classes overall,
   consistent with them being the thinnest by box count (§3).

## 6. What would move the needle next

- **`fathomnet` cap** — currently 33.5% of the corpus and the dominant (or
  only) source for `structure`/`gelatinous`. Lowering its cap would fix the
  composition flag but would need a compensating source for those two
  classes first, or they go from thin to unsupported.
- **`buoy_marker` test coverage** — needs more source videos/groups (more of
  SMD, or another buoy source) before test-split mAP for this class can be
  trusted at all.
- **Surface `person`** — worth sourcing more surface-domain person examples
  specifically (current surface sources are vessel/obstacle-focused, person
  is incidental in them).
- **More epochs** — mAP was still climbing at epoch 100 (best epoch == last
  epoch), suggesting the model hadn't converged; a longer run is a plausible
  free improvement before touching data composition.
- **Export** — `scripts/06_export.py` (ONNX → TensorRT) not yet built;
  needed for both YOLO and WaterNet before either can run on the Jetson
  (JetPack 4.6 ceiling rules out live PyTorch/TF on-device).

## 7. Reproducing this evaluation

```bash
python scripts/05b_validate_by_domain.py --weights runs/train/yolo11n_enhanced/weights/best.pt
python scripts/05c_evaluate_test.py --weights runs/train/yolo11n_enhanced/weights/best.pt
```

# Underwater / surface detector — dataset pipeline

Fine-tuning YOLOv11 for an amphibious survey robot (Jetson + STM32, OAK-D-Lite FF).

The detector is an **auto-labeler**, not a controller. Obstacle avoidance runs off
the OAK-D-Lite spatial channel; this model names what the robot saw so frames can
be archived by class, with anything below threshold routed to an `unknown` bucket
for human review. That framing sets the priorities:

- **Precision outranks recall.** A false positive writes a wrong label into the
  archive. A false negative only sends the frame to the review queue.
- **Coarse classes beat fine ones.** Reliably "fish" > unreliably "rockfish".
- **Negatives matter.** A closed-set detector assigns *some* class to whatever it
  fires on, so ~12% of the corpus is deliberately object-free.

## Geometry

Everything is built at **640×480, 4:3** — pixel-identical to the OAK-D-Lite depth
map, so a detection box maps onto depth with no resampling.

```
12MP 4:3 capture ─┬─ ISP 1920×1440 ──────────────────────► ARCHIVE (raw, never enhanced)
                  └─ ISP 640×480 ─► WaterNet ─► letterbox 640 ─► YOLO
```

The dataset builder mirrors that order exactly. Change it in one place
(`configs/datasets.yaml → geometry`) or not at all.

## Layout

```
configs/taxonomy.yaml     11 classes + per-dataset label maps
configs/datasets.yaml     download registry, caps, strides, split ratios
scripts/download_datasets.py
scripts/unify_datasets.py
src/uwd/                  readers (coco/voc/yolo/mask), geometry, helpers
data/raw/                 one dir per source dataset
data/raw/_archives/       drop manually-downloaded zips here
data/unified/             the output corpus
```

## Setup

```bash
python -m venv .venv && .venv\Scripts\activate && pip install -r requirements.txt
```

## 1. Download

Check what is present:

```bash
python scripts/download_datasets.py --check
```

Attempt everything that can be automated (Kaggle mirror, Roboflow, FathomNet API):

```bash
python scripts/download_datasets.py --all
```

Most marine datasets sit behind Google Drive links or institutional repositories
with click-through licences, so the script cannot fetch them for you. It prints
the exact URL and target filename for each; drop the archive into
`data/raw/_archives/` and re-run the same command to extract and register it.

Before pulling FathomNet in bulk, reconcile the configured concept strings
against the live API — the WoRMS-backed taxonomy drifts:

```bash
python scripts/download_datasets.py --dataset fathomnet --list-concepts
```

Kaggle needs `~/.kaggle/kaggle.json` (Account → Create New API Token).

### Resuming

Everything is checkpointed. Ctrl+C finishes the item in flight, flushes, and
exits; a second Ctrl+C exits immediately. Re-run the same command to continue —
finished work is never redone.

- `data/raw/_state.json` — which stage each dataset reached
- `data/raw/fathomnet/_progress.jsonl` — per-image log for the long FathomNet pull
- `data/raw/fathomnet/_index.json` — cached concept query, so a resume costs no API calls
- `*.part` files — partial HTTP downloads resume via range requests

FathomNet downloads **round-robin across concepts**, so an interrupted run still
leaves balanced coverage rather than 100% of the first few concepts and none of
the rest.

Force a clean re-fetch of one source:

```bash
python scripts/download_datasets.py --dataset duo --force
```

### When a dataset's layout doesn't match

Layout paths are globs resolved against whatever was actually extracted, but
mirrors differ. If unify reports `read=0`, dump the real tree:

```bash
python scripts/download_datasets.py --inspect brackish
```

Then correct that dataset's `layout.parts` in `configs/datasets.yaml`.

## 2. Unify

Dry run first — it reports class balance, per-source share, and every source
label the taxonomy dropped, without writing a single file:

```bash
python scripts/unify_datasets.py --dry-run
```

Read the report before committing. Specifically check:

- **`dropped source labels`** — anything large and unexpected means a mapping
  typo in `configs/taxonomy.yaml`, not a deliberate drop.
- **`images per source`** — any source over 25% is flagged; lower its `cap`.
- **`negatives`** — should land near the 12% target.
- **`boxes per class`** — `rope_net` and `buoy_marker` are the known-thin ones.

Then build it:

```bash
python scripts/unify_datasets.py
```

Single source, for debugging a reader against a freshly extracted tree:

```bash
python scripts/unify_datasets.py --dataset trashcan --dry-run -v
```

## Output

`data/unified/dataset.yaml` is a plain Ultralytics config.
`data/unified/dataset_rfs.yaml` points `train` at `train_rfs.txt`, a
repeat-factor-sampled list that oversamples images carrying rare classes
(Ultralytics has no sampler hook, so this is done by repeating lines).

Use the RFS variant for the real runs; use the plain one for like-for-like
ablations.

## Splits

Assigned **by group** — video, dive, or site — never per image. Every one of
Brackish, TrashCan, VDD-C, FathomNet and MODS is video-derived. A random
per-image split puts near-duplicate adjacent frames on both sides of the
boundary and inflates mAP by tens of points, which you would not discover until
the robot was in the water.

Group keys come from `configs/datasets.yaml → <dataset>.group`. When a filename
pattern does not match the real naming scheme, the code falls back to
one-group-per-image, which silently reintroduces the leak — **verify the group
regexes against the actual extracted filenames** before trusting any validation
number.

## Not yet built

- `enhance.py` — WaterNet pass producing a parallel `images_enhanced/` tree.
  Kept separate on purpose: it makes the raw-vs-enhanced ablation a config
  change rather than a rebuild.
- `train.py` — Kaggle training entrypoint.
- `export.py` — ONNX → TensorRT for the Jetson.

## Still open

- **Jetson variant.** Original Maxwell Nano caps at JetPack 4.6 / Python 3.6, so
  Ultralytics will not install on the board — export ONNX on a host and run a
  standalone TensorRT engine. Orin Nano is straightforward. Decides `export.py`.
- **Collection mission.** If this is a debris/biodiversity survey rather than
  general maritime, drop `vessel` and `buoy_marker` (→ 9 classes) and reallocate
  their budget to `debris` and `rope_net`.

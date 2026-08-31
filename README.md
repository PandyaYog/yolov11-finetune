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
configs/taxonomy.yaml              11 classes + per-dataset label maps
configs/datasets.yaml               download registry, caps, strides, split ratios
scripts/01_download_datasets.py     stage 1: fetch every source into data/raw/
scripts/01_analyze_raw_datasets.py  stage 1: audit an extracted tree (dupes, collisions)
scripts/02_extract_brackish_frames.py  stage 2: decode brackish's raw .avi clips to frames
scripts/02_flatten_vddc_images.py      stage 2: flatten VDD-C's per-session image folders
scripts/02_flatten_vddc_labels.py      stage 2: flatten VDD-C's per-session label folders
scripts/03_unify_datasets.py        stage 3: merge everything into one YOLO corpus
scripts/04_enhance_waternet.py      stage 4: WaterNet pass over submerged-domain images
scripts/05_train.py                 stage 5: fine-tune YOLOv11n
src/uwd/                            readers (coco/voc/yolo/mask), geometry, waternet, helpers
data/raw/                           one dir per source dataset
data/raw/_archives/                 drop manually-downloaded zips here
data/unified/                       stage 3 output: raw-image corpus
data/unified_enhanced/              stage 4 output: WaterNet(submerged) + raw(surface) corpus
models/waternet_checkpoint/         WaterNet TF1 checkpoint (coarse_112/)
runs/train/                         stage 5 output: Ultralytics run dirs, weights/best.pt
```

Stage 2 scripts only apply to the two sources whose raw layout needs a repair
pass before `03_unify_datasets.py` can read them (brackish ships videos, not
frames; VDD-C ships per-session folders, not a flat tree) -- every other
source is read directly out of what stage 1 extracted. Numbers are reserved
past 05 for the pipeline stages in "Not yet built" below.

## Setup

Stages 1-3 and stage 4 (WaterNet) share one venv. Stage 5 (fine-tuning) needs
its own, **separate** venv -- `tensorflow[and-cuda]` (stage 4) and
`ultralytics`'s torch (stage 5) pull different, incompatible cuDNN builds
(`nvidia-cudnn-cu12` vs `nvidia-cudnn-cu13`) that fail at the first real conv
forward pass if installed together:
`CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH`. Two venvs sidesteps it entirely.

```bash
# stages 1-4
python -m venv env && env/bin/pip install -r requirements.txt

# stage 5 (separate venv -- see above)
python -m venv env_train && env_train/bin/pip install -r requirements-train.txt
```

On Kaggle, skip `requirements-train.txt` entirely -- its notebooks ship a
torch already built against their T4 drivers; just `pip install ultralytics`
on top of it.

## 1. Download

Check what is present:

```bash
python scripts/01_download_datasets.py --check
```

Attempt everything that can be automated (Kaggle mirror, Roboflow, FathomNet API):

```bash
python scripts/01_download_datasets.py --all
```

Most marine datasets sit behind Google Drive links or institutional repositories
with click-through licences, so the script cannot fetch them for you. It prints
the exact URL and target filename for each; drop the archive into
`data/raw/_archives/` and re-run the same command to extract and register it.

Before pulling FathomNet in bulk, reconcile the configured concept strings
against the live API — the WoRMS-backed taxonomy drifts:

```bash
python scripts/01_download_datasets.py --dataset fathomnet --list-concepts
```

Kaggle needs `~/.kaggle/kaggle.json` (Account → Create New API Token).

Audit an extracted tree for duplicate/colliding filenames (useful after
dropping in a manual archive):

```bash
python scripts/01_analyze_raw_datasets.py
```

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
python scripts/01_download_datasets.py --dataset duo --force
```

### When a dataset's layout doesn't match

Layout paths are globs resolved against whatever was actually extracted, but
mirrors differ. If unify reports `read=0`, dump the real tree:

```bash
python scripts/01_download_datasets.py --inspect brackish
```

Then correct that dataset's `layout.parts` in `configs/datasets.yaml`.

## 2. Prepare

Two sources extract into a raw layout `03_unify_datasets.py` can't read
directly and need a one-time repair pass first:

```bash
python scripts/02_extract_brackish_frames.py   # decodes dataset/videos/*.avi -> images_extracted/ (needs ffmpeg)
python scripts/02_flatten_vddc_images.py       # per-session folders -> images_flat/
python scripts/02_flatten_vddc_labels.py       # per-session folders -> labels_flat/
```

All three are safe to re-run: each skips or no-ops on anything already moved
or decoded. Every other source is read straight out of what stage 1 extracted.

## 3. Unify

Dry run first — it reports class balance, per-source share, and every source
label the taxonomy dropped, without writing a single file:

```bash
python scripts/03_unify_datasets.py --dry-run
```

Read the report before committing. Specifically check:

- **`dropped source labels`** — anything large and unexpected means a mapping
  typo in `configs/taxonomy.yaml`, not a deliberate drop.
- **`images per source`** — any source over 25% is flagged; lower its `cap`.
- **`negatives`** — should land near the 12% target.
- **`boxes per class`** — `rope_net` and `buoy_marker` are the known-thin ones.

Then build it:

```bash
python scripts/03_unify_datasets.py
```

Single source, for debugging a reader against a freshly extracted tree:

```bash
python scripts/03_unify_datasets.py --dataset trashcan --dry-run -v
```

## 4. Enhance

The robot only runs WaterNet when the depth sensor says it's submerged --
on the surface there's no underwater light attenuation to correct, so running
it there would distort a perfectly good image instead of fixing one.
Training has to match that split or the model gets fine-tuned on a
distribution it never sees live. `configs/datasets.yaml` tags every source
`domain: submerged` or `domain: surface`; this stage runs WaterNet only on
the submerged ones and copies the surface ones through untouched:

```bash
python scripts/04_enhance_waternet.py --dry-run   # counts only, loads nothing
python scripts/04_enhance_waternet.py
```

Output is `data/unified_enhanced/` -- a **separate** corpus root, not a
subdirectory of `data/unified/`, laid out identically
(`images/{train,val,test}`, `labels/{train,val,test}`, `dataset.yaml`,
`dataset_rfs.yaml`). This isn't a style choice: Ultralytics finds an image's
label file by textually replacing `/images/` with `/labels/` in its path, so
a sibling `images_enhanced/` tree wouldn't resolve to any labels at all --
training would silently see zero boxes for every image. `labels/` here is a
copy of `data/unified/labels/` (enhancement doesn't move boxes).

Needs `models/waternet_checkpoint/coarse_112/` (the released Water-Net
checkpoint -- https://github.com/Li-Chongyi/Water-Net_Code) and
`tensorflow[and-cuda]` (see `requirements.txt`; the checkpoint is TF1-era,
loaded here via `tf.compat.v1` in `src/uwd/waternet.py`). It's a real network
pass per image (~0.16s/image on a discrete GPU, ~1.2s/image CPU-only) --
worth confirming a GPU is actually picked up before running it over the full
submerged slice; `04_enhance_waternet.py` logs which device it lands on.

## 5. Train

Fine-tunes YOLOv11n against `data/unified_enhanced/dataset_rfs.yaml` by
default -- the enhanced + repeat-factor-sampled corpus, which is what should
go into the real run. Needs the `env_train` venv (see "Setup").

Local smoke test first -- confirms the dataset format, label discovery, and
GPU all actually work together, in under a minute, before spending real time
on it:

```bash
env_train/bin/python scripts/05_train.py --fraction 0.02 --epochs 1 --batch 8
```

Watch for `train: Scanning .../labels/train.cache... N images, 0 corrupt` --
if labels resolve to 0 or every image comes back a background, the corpus
layout broke (see the note under "4. Enhance" above about why `images_`/`labels_`
naming matters here).

Real run:

```bash
env_train/bin/python scripts/05_train.py --epochs 100
```

### Kaggle

The repo (code) and the corpus (data) travel to Kaggle separately -- don't
try to git-push `data/`, it's gigabytes and doesn't belong in the repo.

**What to upload, as a Kaggle Dataset:** the *contents* of
`data/unified_enhanced/` -- `images/`, `labels/`, `dataset.yaml`,
`dataset_rfs.yaml`, `train_rfs.txt` (~2.3GB). Not `data/unified/` (that's
only for the raw-vs-enhanced ablation, skip it unless you're specifically
running that), and not `models/waternet_checkpoint/` (that was only needed
to *build* the enhanced corpus, already done -- Kaggle trains YOLO directly
against the images that are already enhanced).

**Dataset name:** `uwd-unified-enhanced` (matches this repo's package name,
`src/uwd/`). Whatever you actually name it, `05_train.py` doesn't need to
know in advance -- see below.

1. On kaggle.com/datasets → New Dataset, drag in `data/unified_enhanced/`'s
   contents (or the whole folder -- either upload shape works, see below),
   name it `uwd-unified-enhanced`, create it.
2. New Notebook → Add Input → attach that dataset. Turn on a GPU (T4 x2) under
   Settings → Accelerator.
3. In a notebook cell:
   ```bash
   !git clone <your-repo-url> repo
   %cd repo
   !pip install -q ultralytics
   !python scripts/05_train.py --epochs 100 --device 0,1 --batch 64
   ```

No `--data` flag needed: `05_train.py` first looks for
`data/unified_enhanced/dataset_rfs.yaml` inside the cloned repo (won't exist
on Kaggle, since data/ isn't part of the git push), then falls back to
searching `/kaggle/input/*/` for a `dataset_rfs.yaml` -- which finds it
regardless of whether Kaggle mounted your upload flat
(`/kaggle/input/uwd-unified-enhanced/dataset_rfs.yaml`) or nested under an
extra folder (`/kaggle/input/uwd-unified-enhanced/unified_enhanced/dataset_rfs.yaml`),
since dataset upload UIs are inconsistent about which one you get. Only pass
`--data` explicitly if you've attached more than one dataset containing a
`dataset_rfs.yaml` (the script will tell you if it can't disambiguate).

Kaggle sessions have a wall-clock limit; resume an interrupted run with
`--resume --name <same --name as before>`. For a like-for-like raw-vs-enhanced
ablation, point `--data` at `data/unified/dataset_rfs.yaml` instead (upload
`data/unified/` as a second Kaggle dataset first).

## Output

`dataset.yaml` (in both `data/unified/` and `data/unified_enhanced/`) is a
plain Ultralytics config. `dataset_rfs.yaml` points `train` at
`train_rfs.txt`, a repeat-factor-sampled list that oversamples images
carrying rare classes (Ultralytics has no sampler hook, so this is done by
repeating lines).

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

- `scripts/06_export.py` — ONNX → TensorRT for the Jetson. WaterNet needs the
  same export treatment as YOLO here, not just the detector -- Jetson Nano's
  JetPack 4.6 ceiling means neither can run as live TF/PyTorch on the bot.

## Still open

- **Jetson variant.** Original Maxwell Nano caps at JetPack 4.6 / Python 3.6, so
  Ultralytics will not install on the board — export ONNX on a host and run a
  standalone TensorRT engine. Orin Nano is straightforward. Decides `export.py`.
- **Collection mission.** If this is a debris/biodiversity survey rather than
  general maritime, drop `vessel` and `buoy_marker` (→ 9 classes) and reallocate
  their budget to `debris` and `rope_net`.

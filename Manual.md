## 1  Repository layout – what each piece does

| Path                                             | Role in the pipeline                                                                                                                                                                                                                                                                                                     |
| ------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `main.py`                                        | Entry‑point. Builds dataloaders, parses CLI flags, instantiates the dual‑encoder model and launches **three** possible training loops: <br>• self‑supervised **Image→Coord** (vectorization) <br>• self‑supervised **Coord→Image** (rasterization) <br>• fully‑supervised linear‑head finetuning (default) ([GitHub][1]) |
| `Dataset.py`                                     | All dataset code. Two `torch.utils.data.Dataset` subclasses: `Dataset_TUBerlin` (pickled vectors) and `Dataset_Quickdraw` (LMDB) plus a common `collate_self` and `get_dataloader()` helper. ([GitHub][2])                                                                                                               |
| `rasterize.py`                                   | Vector‑to‑bitmap renderer (Pure‑PIL/Numpy). Used both at train‑time (to make paired image‑vector data) and for qualitative visualisation. ([GitHub][3])                                                                                                                                                                  |
| `models_Image2Coord.py`, `models_Coord2Image.py` | Translators that wrap the shared encoders/decoders in `Networks.py` and provide the loss heads for the two pre‑text tasks.                                                                                                                                                                                               |
| `Networks.py`                                    | Building blocks: ResNet‑50 image encoder, bidirectional LSTM stroke encoder, U‑Net and residual up‑blocks for the two decoders, plus helper extractors. ([GitHub][4])                                                                                                                                                    |
| `utils.py`                                       | Misc. helpers – stroke‑format conversions, pretty drawing utilities, logging visuals, etc. ([GitHub][5])                                                                                                                                                                                                                 |
| `baselines/`                                     | Reference checkpoints & quick evaluation scripts for TU‑Berlin and QuickDraw (optional).                                                                                                                                                                                                                                 |
| `sample_images/`                                 | Figures that appear in the README › outline/architecture.                                                                                                                                                                                                                                                                |
| `index.html`, `README.md`, `LICENSE`             | Paper teaser site, short README and Apache‑2.0 licence. ([GitHub][6])                                                                                                                                                                                                                                                    |

---

## 2  Environment setup

```bash
# 1.  Create an isolated env (Python ≥3.8, CUDA‑11.x)
conda create -n sketch2vec python=3.9 -y
conda activate sketch2vec

# 2.  Core libraries
pip install torch torchvision torchaudio
# 3.  Repo dependencies
pip install lmdb pillow matplotlib tqdm numpy gdown bresenham scipy
# 4.  (Optional) Jupyter & tensorboard
pip install jupyterlab tensorboard
```

> **GPU memory** Vectorisation & rasterisation both hold the entire ResNet‑50 + 2‑layer Bi‑LSTM + U‑Net in memory. With the default `--batchsize 64` you will need \~11 GB. Drop to 32/16 on smaller cards.

---

## 3  Dataset preparation

### 3.1 TU‑Berlin‑250 (vector pickle, 250 categories)

The repo expects **one pickle file** (`./TU_Berlin`) that maps
`"class_name/img_000123.npy"  →  N×3 numpy array`
(x, y, pen‑state).

```bash
# Grab the author‑provided pickle (~230 MB)
gdown 1pnqwM-i9dVKIkV09X-5Oi8lu_ejpSHvx -O TU_Berlin
```

*No further preprocessing is needed.*
At runtime `Dataset_TUBerlin` performs a 70 / 30 split controlled by `--splitTrain` (default 0.7). ([GitHub][2])

### 3.2 QuickDraw‑345 (LMDB)

1. **Download Google’s raw `.npz` files** for the 345 classes you need:

    ```
    curl -L \
    https://raw.githubusercontent.com/googlecreativelab/quickdraw-dataset/master/categories.txt \
    -o categories.txt
    # Method A
    mkdir -p raw_npz      #  ← folder for the .npz files

    while read -r cls; do
    url="https://storage.googleapis.com/quickdraw_dataset/sketchrnn/{cls}.npz"
    echo "⇣  $cls"
    wget -nc -q --show-progress -P raw_npz "$url"
    done < categories.txt

    # Method B (Parallel)

    parallel --jobs 8 --bar \
    >   'wget -nc -q --show-progress -P raw_npz \
    >      https://storage.googleapis.com/quickdraw_dataset/sketchrnn/{}.npz' \
    >   :::: categories.txt


    ```

2. **Convert to LMDB** (one key per sample) and build a train/val/test key list.

```python
python quickdraw_make_lmdb.py
```

3. Verify the folder tree looks like:

```
QuickDraw/
 ├── QuickDraw_TrainData   (LMDB dir, ~30 GB)
 ├── QuickDraw_TestData    (LMDB dir)
 └── QuickDraw_Keys.pickle
```

`Dataset_Quickdraw` will now work out‑of‑the‑box. ([GitHub][2])

---

## 4  Running the code

### 4.1 Self‑supervised pre‑training (Vectorization)

```bash
python main.py \
  --dataset_name TUBerlin \
  --train_mode supervised \
  --eval_freq_iter 48000
# Uncomment the Image2Coordinate loop in main.py (lines 28‑52) and
# comment out the “Fully Supervised” block. :contentReference[oaicite:8]{index=8}
```

### 4.2 Self‑supervised pre‑training (Rasterization)

Same as above, but flip the comment blocks so the **Coordinate2Image** loop is active.

### 4.3 Linear evaluation

After either self‑supervised run finishes, re‑enable the “Fully Supervised” section (or run in a fresh clone) with

```bash
python main.py --dataset_name TUBerlin --fullysupervised True
```

and point `--backbone_name` / `--pool_method` as needed.

Checkpoints and qualitative drawings are written to
`./results/<dataset>/<date>/models/` and `…/sketch_Viz/`.

---

## 5  Things worth knowing

* **Pickle size filter** – strokes longer than 300 points are silently discarded (`Dataset_TUBerlin`). ([GitHub][2])
* **Horizontal flip augmentation** is done *both* on the raster image and x‑coordinates (`F.hflip` + `x = -x + 256`). ([GitHub][2])
* **Stroke‑5 format** – internal representation matches the sketch‑rnn paper (Δx, Δy, p1, p2, p3). `utils.to_Five_Point` and `to_normal_strokes` handle conversions. ([GitHub][5])
* **Encoder weight reuse** – for downstream tasks you can load `Resnet_Network` or `Sketch_LSTM` weights from the self‑supervised checkpoint and drop the decoders.
* **Multi‑GPU** – the repo has no `DistributedDataParallel`; use `CUDA_VISIBLE_DEVICES=0,1` and wrap `model = torch.nn.DataParallel(model)` manually if desired.
* **Paper hyper‑parameters** – Table 1 in the CVPR paper uses **batch 64**, **SGD** (lr 1e‑4) for 1 M iterations on QuickDraw‑50 M; reproducing exactly will require larger epoch/iter counts than the default 50 epochs in `main.py`.
* **Baselines** – the `baselines/` folder contains checkpoints reproducing the paper’s linear‑eval numbers; handy for sanity tests on your data pre‑processing.

Enjoy hacking!

[1]: https://github.com/AyanKumarBhunia/Self-Supervised-Learning-for-Sketch/raw/main/main.py?plain=1 "github.com"
[2]: https://github.com/AyanKumarBhunia/Self-Supervised-Learning-for-Sketch/raw/main/Dataset.py?plain=1 "github.com"
[3]: https://github.com/AyanKumarBhunia/Self-Supervised-Learning-for-Sketch/blob/main/rasterize.py?utm_source=chatgpt.com "Self-Supervised-Learning-for-Sketch/rasterize.py at main - GitHub"
[4]: https://github.com/AyanKumarBhunia/Self-Supervised-Learning-for-Sketch/raw/main/Networks.py?plain=1 "github.com"
[5]: https://github.com/AyanKumarBhunia/Self-Supervised-Learning-for-Sketch/raw/main/utils.py?plain=1 "github.com"
[6]: https://github.com/AyanKumarBhunia/Self-Supervised-Learning-for-Sketch/raw/main/README.md?plain=1 "github.com"

# ── vectorise_tuberlin.py ──────────────────────────────────────────
import cv2, numpy as np, os, pickle, tqdm
root = "TU_raw"; all_vec = {}
for cls in tqdm.tqdm(sorted(os.listdir(root))):
    for f in os.listdir(f"{root}/{cls}"):
        g = cv2.imread(f"{root}/{cls}/{f}", 0)
        g = cv2.resize(g, (256,256), cv2.INTER_NEAREST)
        _, th = cv2.threshold(g, 127, 255, cv2.THRESH_BINARY_INV)
        cnts,_ = cv2.findContours(th, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        strokes = []
        for c in cnts:
            pts = c.squeeze(1).astype(np.float32) - 128        # centre
            dxy  = np.diff(pts, axis=0, prepend=pts[:1])
            pen  = np.zeros((len(pts),1), np.float32); pen[-1]=1
            strokes.append(np.hstack([dxy, pen]))
        if strokes:
            all_vec[f"{cls}/{f}"] = np.concatenate(strokes)
with open("TU_Berlin", "wb") as fp: pickle.dump(all_vec, fp, pickle.HIGHEST_PROTOCOL)
# ───────────────────────────────────────────────────────────────────

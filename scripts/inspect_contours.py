import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "src"))

from src.io_utils import load_contours_npz


def extract_contours(obj):
    if isinstance(obj, (list, tuple)):
        return [np.asarray(c) for c in obj]

    if hasattr(obj, "keys"):
        keys = list(obj.keys())

        if "contours" in keys:
            arr = obj["contours"]
            arr = np.asarray(arr, dtype=object)
            return [np.asarray(c) for c in arr]

        contour_keys = sorted([k for k in keys if str(k).startswith("contour_")])
        return [np.asarray(obj[k]) for k in contour_keys]

    raise TypeError(type(obj))


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--contours_npz", required=True)
    args = parser.parse_args()

    img = np.array(Image.open(args.image).convert("RGB"))

    loaded = load_contours_npz(args.contours_npz)
    contours = extract_contours(loaded)

    fig, ax = plt.subplots(figsize=(10,8))
    ax.imshow(img)

    for i, c in enumerate(contours):
        ax.plot(c[:,0], c[:,1], linewidth=2)
        mid = c[len(c)//2]
        ax.text(mid[0], mid[1], str(i), color="yellow", fontsize=12)

    ax.set_title(f"{len(contours)} contours detected")
    ax.axis("off")

    plt.show()


if __name__ == "__main__":
    main()

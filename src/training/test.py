import sys
from pathlib import Path
import cv2
import numpy as np


direct = Path(sys.argv[1]).resolve()


for path in direct.iterdir():
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    img = img.astype(np.uint16)

    assert img.dtype == np.uint16
    assert len(img.shape) == 2

    img_min, img_max = img.min(), img.max()

    if img_max > img_min:
        img = (img - img_min) / (img_max - img_min) * 65535.0

    img = np.round(img).astype(np.uint16)

    cv2.imwrite(path, img, [cv2.IMWRITE_PNG_COMPRESSION, 0])

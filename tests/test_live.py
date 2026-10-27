import base64
import copy
import io
import json
import struct
import time
import zipfile
from pathlib import Path
from threading import Lock

import cv2
import numpy as np
import pytest

from opencv_deploy_proof import live, pedestrian
from opencv_deploy_proof.data import load_manifest


def test_images_are_bounded_before_decode_and_jpeg_is_supported():
    image = np.zeros((12, 16), np.uint8)
    for extension in ('.png', '.jpg'):
        _, raw = cv2.imencode(extension, image)
        assert live.decode_image(raw.tobytes()).shape == (12, 16)
    huge = b'\x89PNG\r\n\x1a\n' + struct.pack('>I', 13) + b'IHDR' + struct.pack('>II', 100000, 100000)
    with pytest.raises(ValueError, match='megapixels'):
        live.decode_image(huge)
    with pytest.raises(ValueError):
        live.decode_base64('invalid===', 100)

"""Small, classical OpenCV defect-localization pipeline for aligned image pairs."""

from dataclasses import asdict, dataclass
from time import perf_counter_ns

import cv2
import numpy as np


@dataclass(frozen=True)
class Config:
    scale: float = 1.0
    registration_scale: float = 1.0
    threshold: int = 128
    min_area: int = 9
    open_kernel: int = 3
    workers: int = 1
    threads: int = 1

    def __post_init__(self):
        if self.scale not in (0.5, 0.75, 1.0):
            raise ValueError("scale must be 0.5, 0.75 or 1.0")
        if self.registration_scale not in (0.25, 0.5, 1.0):
            raise ValueError("registration_scale must be 0.25, 0.5 or 1.0")
        if not 10 <= self.threshold <= 240 or not 1 <= self.min_area <= 100:
            raise ValueError("invalid threshold/min_area")
        if self.open_kernel not in (1, 3) or self.workers not in (1, 2, 4) or self.threads not in (1, 2, 4):
            raise ValueError("invalid kernel/workers/threads")
        if self.workers > 1 and self.threads > 1:
            raise ValueError("nested thread pools are excluded from the experiment space")

    def to_dict(self):
        return asdict(self)


def detect(image: np.ndarray, template: np.ndarray, config: Config) -> dict:
    if image is None or template is None or image.shape != template.shape or image.ndim != 2:
        raise ValueError("expected an aligned pair of grayscale images with identical dimensions")
    stages = {}
    start = perf_counter_ns()
    # The benchmark's 'aligned' crops still contain subpixel translation jitter.
    # Estimate translation from image evidence; annotation boxes never enter this path.
    alignment_shape = (max(16, round(image.shape[1] * config.registration_scale)),
                       max(16, round(image.shape[0] * config.registration_scale)))
    small_image = cv2.resize(image, alignment_shape, interpolation=cv2.INTER_AREA)
    small_template = cv2.resize(template, alignment_shape, interpolation=cv2.INTER_AREA)
    shift, response = cv2.phaseCorrelate(np.float32(small_template), np.float32(small_image))
    shift = (shift[0] * image.shape[1] / alignment_shape[0],
             shift[1] * image.shape[0] / alignment_shape[1])
    if response > 0.1 and max(abs(shift[0]), abs(shift[1])) < 12:
        warp = np.float32([[1, 0, shift[0]], [0, 1, shift[1]]])
        template = cv2.warpAffine(template, warp, image.shape[::-1], borderMode=cv2.BORDER_REPLICATE)
    stages["registration"] = (perf_counter_ns() - start) / 1e6
    start = perf_counter_ns()
    if config.scale != 1:
        shape = (round(image.shape[1] * config.scale), round(image.shape[0] * config.scale))
        image = cv2.resize(image, shape, interpolation=cv2.INTER_AREA)
        template = cv2.resize(template, shape, interpolation=cv2.INTER_AREA)
    stages["resize"] = (perf_counter_ns() - start) / 1e6
    start = perf_counter_ns()
    # Binarization avoids JPEG/intensity noise in the already registered benchmark pairs.
    _, binary = cv2.threshold(image, config.threshold, 255, cv2.THRESH_BINARY)
    _, reference = cv2.threshold(template, config.threshold, 255, cv2.THRESH_BINARY)
    tolerance = np.ones((3, 3), np.uint8)
    # Symmetric one-pixel tolerance prevents edge jitter becoming a field of false defects.
    difference = cv2.bitwise_or(
        cv2.bitwise_and(binary, cv2.bitwise_not(cv2.dilate(reference, tolerance))),
        cv2.bitwise_and(reference, cv2.bitwise_not(cv2.dilate(binary, tolerance))))
    stages["threshold_difference"] = (perf_counter_ns() - start) / 1e6
    start = perf_counter_ns()
    if config.open_kernel > 1:
        kernel = np.ones((config.open_kernel, config.open_kernel), np.uint8)
        difference = cv2.morphologyEx(difference, cv2.MORPH_OPEN, kernel)
    padding = max(3, round(13 * config.scale) | 1)
    difference = cv2.dilate(difference, np.ones((padding, padding), np.uint8))
    stages["morphology"] = (perf_counter_ns() - start) / 1e6
    start = perf_counter_ns()
    contours, _ = cv2.findContours(difference, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        if cv2.contourArea(contour) >= config.min_area * config.scale ** 2:
            boxes.append([round(x / config.scale), round(y / config.scale),
                          round((x + width) / config.scale), round((y + height) / config.scale)])
    stages["contours"] = (perf_counter_ns() - start) / 1e6
    return {"boxes": sorted(boxes), "stages_ms": stages}

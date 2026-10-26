# Third-party attribution

The pedestrian adapter uses OpenCV Zoo's YOLOX model and follows its reference
grid/stride decoder. OpenCV Zoo and the YOLOX model folder carry the Apache License 2.0.
The upstream source is pinned to commit `48b10fa505aeb51fbd127fc60a5c660040e823b7`:
https://github.com/opencv/opencv_zoo/tree/48b10fa505aeb51fbd127fc60a5c660040e823b7/models/object_detection_yolox

Model weights are downloaded separately, checked against SHA256, and are not included
in this repository. The complete upstream license is in `licenses/YOLOX-Apache-2.0.txt`.

Penn-Fudan images and instance masks come from the original research dataset by
Liming Wang, Jianbo Shi, Gang Song and I-fan Shen:
https://www.cis.upenn.edu/~jshi/ped_html/
Dataset images are downloaded separately. No ownership or new license over those
images is claimed. Derived boxes retain their mask provenance in the manifest.

DeepPCB is used under its MIT license from the pinned upstream dataset. The download
retains its LICENSE file. No dataset images or model weights are bundled in source exports.

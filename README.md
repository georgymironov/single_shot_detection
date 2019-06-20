# Single Shot Detection
Build flexible object detection pipelines with declarative configuration
### Content
You are being provided with the following set of features:
- Supporting latest PyTorch release
- Train SSD, M2Det, RetinaNet or some custom architecture
- Available backbones: torchvision + MobileNet, MobileNetV2
- Data augmentations
- [AdamW and SGDW](https://www.fast.ai/2018/07/02/adam-weight-decay/) optimizers, some custom learning rate schedulers
- Weight pruning for efficient inference
- Export to [ONNX](https://github.com/onnx/onnx) or [OpenVINO](https://github.com/opencv/dldt)
- Tensorboard integration
- Training callbacks

### Quick start
Download [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/) or [COCO](http://cocodataset.org/) dataset and start training using one of the provided sample configs:
```
python3 main.py --config samples/ssd_mb2_voc.py
```
<sup><sup>(don't forget to adjust paths in the config first!)</sup></sup>

To see the list of parameters:
```
python3 main.py --help
```
### Requirements
- `python 3.6`
- `opencv` with python bindings
- `libturbojpeg`
- `requirements.txt`
### Structure
Some places that may be useful to look into:
- `bf` - provides common reusable parts for building a deep learning pipeline
    - `bf.base` - custom backbone network implementation (e.g. MobileNetV2)
    - `bf.datasets` - dataset handling
    - `bf.preprocessing` - data augmentations and preprocessing
    - `bf.training` - callbacks; custom optimizers and learning
    rate schedulers; weight prunner
    - ...
- `detection` - parts of code which are used to build object detection pipelines on top of `bf`
- `samples` - contains sample configuration files for popular network architectures
- `main.py` - the entry point
### Inspired by
- Tensorflow Object Detection API (https://github.com/tensorflow/models/tree/master/research/object_detection)
- Gluon CV (https://github.com/dmlc/gluon-cv)
- mmdetection (https://github.com/open-mmlab/mmdetection)

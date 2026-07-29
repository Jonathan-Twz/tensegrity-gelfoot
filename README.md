<p align="center">
<!--   <img src="assets/figs/unidrive-logo.png" align="center" width="20%"> -->
  
  <h3 align="center"><strong>Scalable Open-Source Visuotactile Sensor for 6-Axis Contact Wrench Estimation in Tensegrity Robots</strong></h3>

  <p align="center">
      <a href="https://jonathan-twz.github.io/" target='_blank'>Wenzhe Tong*</a>&nbsp;&nbsp;&nbsp;
      <a href="https://www.linkedin.com/in/jonathanmi6/" target='_blank'>Jonathan Mi*</a>&nbsp;&nbsp;&nbsp;
      <a href="https://eclymk.github.io/xili-web/" target='_blank'>Xili Yi</a>&nbsp;&nbsp;&nbsp;
      <a href="https://www.mmintlab.com/people/nima-fazeli/" target='_blank'>Nima Fazeli</a>&nbsp;&nbsp;&nbsp;
      <a href="https://robotics.umich.edu/profile/xiaonan-sean-huang/" target='_blank'>Xiaonan Huang</a>
      <br />
  </p>

  <p align="center">
    <img src="assets/u-m_logo-horizontal-hex.png" height="40">
  </p>
   <h3 align="center"> IROS 2026 </h3>
</p>

<p align="center">
  <a href="https://arxiv.org/pdf/2607.15633" target='_blank'>
    <img src="https://img.shields.io/badge/Paper-%F0%9F%93%83-slategray">
  </a>

  <a href="https://github.com/Jonathan-Twz/tensegrity-gelfoot" target='_blank'>
    <img src="https://img.shields.io/badge/Code-%F0%9F%94%97-lightblue">
  </a>

  <a href="https://youtu.be/4Ih4xmbAdz4" target='_blank'>
    <img src="https://img.shields.io/badge/Video-%F0%9F%8E%AC-pink">
  </a>
  
  <!-- <a href="" target='_blank'>
    <img src="https://img.shields.io/badge/%E4%B8%AD%E8%AF%91%E7%89%88-%F0%9F%90%BC-red">
  </a> -->
  
  <a href="" target='_blank'>
    <img src="https://visitor-badge.laobi.icu/badge?page_id=Jonathan-Twz.tensegrity-gelfoot&left_color=gray&right_color=firebrick">
  </a>
</p>


# Tensegrity Gelfoot

Gelfoot combines an elastomeric sensing surface, a camera and illumination
module, dense optical-flow-based shear estimation, and a residual MLP that maps
a two-channel shear field to force and torque.


<!-- ## System overview

```text
camera image
    -> crop and resize
    -> GelSlim shear-field estimation
    -> 2 x 30 x 30 vector field
    -> residual MLP
    -> 6D wrench
    -> contact detection / ROS topics
``` -->

### Repository layout

```text
.
├── checkpoints/                  pretrained model weights
├── data/                         dataset folder, download from huggingface
├── hardware/                     STEP files for gelfoot
├── scripts/
│   ├── ROS_wrench_inference.py   1 cam runner 
│   ├── run_6cam_pipeline.py      6 cams runner
│   ├── record_rosbag.sh          
│   └── 6cam.rviz                 
├── src/
│   ├── arm_control/              KUKA & ATI F/T ROS nodes, data collection
│   └── gelslim_shear/            shear-field estimation
└── train.ipynb                   
```

## Installation

```bash
conda create --name gelfoot python=3.13.5 pip
conda activate gelfoot
pip install -r requirements.txt
```

The requirements file also installs `src/gelslim_shear` in editable mode. 

### ROS dependencies (optional)

The real-time and data-collection tools are tested on ROS 1 Noetic, with the following dependencies:

- `rospy`, `cv_bridge`, `rosbag` 
- [`tensegrity_msgs`](https://github.com/UMich-HDRLab/tensegrity_msgs), provides `Float32MultiArrayStamped` message type.
- [`src/arm_control/README.md`](src/arm_control/README.md) contains instructions for installing the KUKA and ATI F/T ROS nodes.

### Dataset (optional)

Download and extract the processed Gelfoot dataset from
[Hugging Face](https://huggingface.co/datasets/jonathantwz/gelfoot):

```bash
curl -L --fail -o data.zip "https://huggingface.co/datasets/jonathantwz/gelfoot/resolve/main/data.zip?download=true"

unzip data.zip
```

## Guidelines

### Standalone visualization (without ROS)

```bash
python3 src/gelslim_shear/run.py
```

The camera index is currently selected in the script. Confirm the correct
`/dev/video*` mapping before running it.

### ROS node

Start `roscore` and, in a shell with the ROS workspace and Python environment
sourced, run:

```bash
python3 src/gelslim_shear/ROS_gelslim_node.py \
  _camera_index:=6 \
  _width:=640 \
  _height:=480
```

### Robot 6-camera pipeline

The multi-camera runner defaults to camera indices `0,2,4,6,8,10`. Test camera availability before starting ROS processing:

```bash
python3 scripts/run_6cam_pipeline.py --dry-run
```

To save one warm-started image from each detected camera:

```bash
python3 scripts/run_6cam_pipeline.py --dry-run --save-img
```

run either the default mapping or a custom set of cameras:

```bash
# default camera indices 0,2,4,6,8,10
python3 scripts/run_6cam_pipeline.py

# camera indices 0,1,2,3,4,5
python3 scripts/run_6cam_pipeline.py --cameras 0,1,2,3,4,5
```

For each mapped endcap `N`, the runner publishes:

| Topic | Description |
| --- | --- |
| `/endcap_N/image/cropped` | Cropped camera image |
| `/endcap_N/image/shear` | Shear visualization |
| `/endcap_N/image/divergence` | Divergence visualization |
| `/endcap_N/array/shear_vector` | Dense shear field |
| `/endcap_N/wrench/wrench_pred` | Predicted six-axis wrench |
| `/endcap_N/wrench/force_norm` | Predicted force magnitude |

Camera-to-endcap assignment is hardware-specific. The mapping currently used by
the robot is defined in `GlobalConfig.CAMERA_TO_ENDCAP` in
[`scripts/run_6cam_pipeline.py`](scripts/run_6cam_pipeline.py).

## Dataset

The dataset recorder captures one synchronized shear/wrench pair for each
rising trigger on `/dataset/is_recording`:

```bash
python3 src/arm_control/helpers/dataset_recorder.py \
  _output_dir:="$PWD/data" \
  _wrench_topic:=/wrench_grasp_frame \
  _shear_topic:=/gelslim/array/shear_vector \
  _trigger_topic:=/dataset/is_recording
```

Approximate timestamp synchronization uses a default tolerance of 50 ms. Each
sample is stored as a NumPy array under `shear_fields/`, with its timestamp,
six-axis wrench, and relative path recorded in `index.csv`.

The provided [dataset](https://huggingface.co/datasets/jonathantwz/gelfoot/tree/main) contains 34,368 samples with shear arrays of shape `(2, 30, 30)`.

## Hardware

The current CAD directory contains STEP models for the arm-side interface and
the sensor endcap base:

```text
hardware/arm-base.step
hardware/endcap-base.step
```

Printable exports, elastomer mold geometry, bill of materials, electronics,
fabrication parameters, and assembly instructions are planned for the public
release.

## Citation
```text
@article{tong2026scalable,
  title={Scalable Open-Source Visuotactile Sensor for 6-Axis Contact Wrench Estimation in Tensegrity Robots},
  author={Tong, Wenzhe and Mi, Jonathan and Yi, Xili and Fazeli, Nima and Huang, Xiaonan},
  journal={arXiv preprint arXiv:2607.15633},
  year={2026}
}
```

## License

This work is under the <a rel="license" href="">MIT License</a>, while the shear-field implementation is derived from the
[MMintLab GelSlim 4.0 shear-field package](https://github.com/MMintLab/gelslim_shear).

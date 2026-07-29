#!/usr/bin/env python3
"""
Multi-camera GelFoot pipeline runner.

Runs 6 parallel camera pipelines:
  Camera -> Shear Processing -> Wrench Inference -> ROS publish

Usage:
    python run_6cam_pipeline.py
    
    # With custom camera indices:
    python run_6cam_pipeline.py --cameras 0,2,4,6,8,10
    
    # Dry run (test cameras only):
    python run_6cam_pipeline.py --dry-run
"""

import os
import sys
import glob
import time
import argparse
import threading
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass
from queue import Queue

# Set environment variable to avoid OpenMP library conflicts
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.feature import peak_local_max

# Add gelslim_shear to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "gelslim_shear"))

from gelslim_shear.shear_utils.shear_from_gelslim import ShearGenerator
from gelslim_shear.plot_utils.shear_plotter import (
    cv_plot_scalar_field,
    cv_plot_vector_field,
    get_channel,
)

# ROS imports
import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import WrenchStamped
from std_msgs.msg import Header, MultiArrayDimension, Float32MultiArray, Float64


def cv2_to_imgmsg(cv_image: np.ndarray, encoding: str = 'bgr8') -> Image:
    """Convert OpenCV image to ROS Image message (replaces cv_bridge)."""
    msg = Image()
    msg.height = cv_image.shape[0]
    msg.width = cv_image.shape[1]
    msg.encoding = encoding
    msg.is_bigendian = 0
    msg.step = cv_image.shape[1] * cv_image.shape[2] if len(cv_image.shape) == 3 else cv_image.shape[1]
    msg.data = cv_image.tobytes()
    return msg

# Try to import custom message, fall back to standard Float32MultiArray
try:
    from tensegrity_msgs.msg import Float32MultiArrayStamped
    HAS_TENSEGRITY_MSGS = True
except ImportError:
    HAS_TENSEGRITY_MSGS = False
    print("Warning: tensegrity_msgs not found, using std_msgs/Float32MultiArray")


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class CameraConfig:
    """Configuration for a single camera pipeline."""
    camera_index: int
    namespace: str
    frame_id: str
    width: int = 640
    height: int = 480


class GlobalConfig:
    """Global configuration constants."""
    FRAME_RATE = 30
    PROCESSED_IMAGE_SIZE = (200, 200)
    SHEAR_OUTPUT_SIZE = (30, 30)
    DISPLAY_SIZE = (600, 600)
    
    MAX_SHEAR_MAGNITUDE = 3.0
    CONTACT_MIN_DISTANCE = 2
    CONTACT_THRESHOLD = 150
    CONTACT_WINDOW_SIZE = 5
    GAUSSIAN_AMPLITUDE_FACTOR = 0.8
    
    MARKER_SIZE = 20
    MARKER_THICKNESS = 5
    SHEAR_COLOR = (255, 0, 0)
    SHEAR_DIFF_COLOR = (0, 0, 255)
    CONTACT_COLOR = (0, 0, 255)
    
    SHEAR_CHANNELS = ['u', 'v', 'div', 'du', 'dv']
    FARNEBACK_PARAMS = (0.5, 3, 45, 3, 5, 1.2, 0)
    
    # Camera USB ID to endcap ID mapping
    # USB ports: [0, 2, 4, 6, 8, 10] -> endcaps: [3, 0, 4, 1, 5, 2]
    CAMERA_TO_ENDCAP = {0: 3,
                        2: 0,
                        4: 4,
                        6: 1,
                        8: 5,
                        10: 2}
    
    # Force threshold for zeroing noise
    FORCE_THRESHOLD = 0.2


# =============================================================================
# Neural Network Model
# =============================================================================

class ResBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(out_features, out_features)
        self.shortcut = nn.Linear(in_features, out_features) if in_features != out_features else nn.Identity()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = out + self.shortcut(x)
        return self.relu(out)


class ResNetRegressor(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 64, num_blocks: int = 2):
        super().__init__()
        layers = [ResBlock(in_dim, hidden_dim)]
        for _ in range(num_blocks - 1):
            layers.append(ResBlock(hidden_dim, hidden_dim))
        self.res_blocks = nn.Sequential(*layers)
        self.fc_out = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.res_blocks(x)
        return self.fc_out(x)


# =============================================================================
# Utility Functions
# =============================================================================

def square_center_crop(image: torch.Tensor) -> torch.Tensor:
    """Crop a 3D tensor to a square by center-cropping the larger dimension."""
    height = image.shape[1]
    width = image.shape[2]
    
    if height > width:
        start = (height - width) // 2
        return image[:, start:start+width, :]
    elif width > height:
        start = (width - height) // 2
        return image[:, :, start:start+height]
    return image


def downsample(image: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    """Downsample an image tensor to the specified size."""
    return F.interpolate(image.unsqueeze(0), size=size, mode='area').squeeze(0)


def center_crop(frame: np.ndarray) -> np.ndarray:
    """Crop a 2D frame to a square by center-cropping the larger dimension."""
    h, w = frame.shape[:2]
    m = min(h, w)
    sy = (h - m) // 2
    sx = (w - m) // 2
    return frame[sy:sy+m, sx:sx+m]


def local_moments(window: np.ndarray, y0: int, x0: int) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate mean and covariance of a patch using weighted second moments."""
    H, W = window.shape
    y, x = np.indices((H, W))
    weights = window
    weights_sum = weights.sum()
    
    if weights_sum == 0:
        return np.array([y0, x0]), np.eye(2)
    
    mu_y = np.sum(y * weights) / weights_sum
    mu_x = np.sum(x * weights) / weights_sum
    
    y_c = y - mu_y
    x_c = x - mu_x
    
    cov_yy = np.sum((y_c ** 2) * weights) / weights_sum
    cov_xx = np.sum((x_c ** 2) * weights) / weights_sum
    cov_xy = np.sum((y_c * x_c) * weights) / weights_sum
    
    cov = np.array([[cov_yy, cov_xy], [cov_xy, cov_xx]])
    cov += 1e-6 * np.eye(2)
    
    return np.array([mu_y + y0, mu_x + x0]), cov


def gaussian_2d(shape: Tuple[int, int], mean: np.ndarray, cov: np.ndarray, amplitude: float = 1.0) -> np.ndarray:
    """Generate a 2D Gaussian distribution over the specified shape."""
    y, x = np.indices(shape)
    pos = np.stack([y - mean[0], x - mean[1]], axis=-1)
    inv_cov = np.linalg.inv(cov)
    exponent = np.einsum('...i,ij,...j->...', pos, inv_cov, pos)
    return amplitude * np.exp(-0.5 * exponent)


def find_checkpoint() -> Optional[str]:
    """Find model checkpoint file."""
    ckpt_dir = os.path.join(SCRIPT_DIR, "checkpoints")
    candidate = os.path.join(ckpt_dir, "wrench_regression_model.pth")
    if os.path.isfile(candidate):
        return candidate
    pths = sorted(glob.glob(os.path.join(ckpt_dir, "*.pth")))
    return pths[0] if pths else None


# =============================================================================
# Camera Pipeline
# =============================================================================

class CameraPipeline:
    """Single camera pipeline: capture -> shear -> inference -> publish."""
    
    def __init__(self, config: CameraConfig, device: torch.device, model: Optional[ResNetRegressor] = None):
        self.config = config
        self.device = device
        self.model = model
        self.frame_id = 0
        self.frame_period = 1.0 / GlobalConfig.FRAME_RATE
        self.running = False
        
        # Initialize shear generator
        self.shgen = ShearGenerator(
            method='2',
            channels=GlobalConfig.SHEAR_CHANNELS,
            Farneback_params=GlobalConfig.FARNEBACK_PARAMS,
            output_size=GlobalConfig.SHEAR_OUTPUT_SIZE
        )
        
        # Load baseline image for this endcap
        endcap_id = GlobalConfig.CAMERA_TO_ENDCAP.get(config.camera_index, 0)
        no_shear_path = os.path.join(SCRIPT_DIR, "gelslim_shear/camera_calibration", f"endcap_{endcap_id}.jpg")
        no_shear_image = cv2.imread(no_shear_path, cv2.IMREAD_COLOR)
        if no_shear_image is None:
            rospy.logwarn(f"Calibration image not found at: {no_shear_path}, using fallback")
            # Try fallback to generic image
            no_shear_path = os.path.join(SCRIPT_DIR, "gelslim_shear/camera_calibration", "no_shear_image.png")
            no_shear_image = cv2.imread(no_shear_path, cv2.IMREAD_COLOR)
            if no_shear_image is None:
                raise RuntimeError(f"No calibration image found for camera {config.camera_index}")
        
        # Convert baseline image to tensor with same preprocessing as runtime frames
        baseline_tensor = torch.from_numpy(no_shear_image).permute(2, 0, 1).float().to(device)
        baseline_tensor = square_center_crop(baseline_tensor)
        baseline_tensor = downsample(baseline_tensor, GlobalConfig.PROCESSED_IMAGE_SIZE)
        self.shgen.update_base_tactile_image(baseline_tensor)
        rospy.loginfo("Loaded baseline image for endcap {} from {}".format(endcap_id, no_shear_path))
        
        # Initialize camera (lazy)
        self.cap = None
        
        # Setup ROS publishers
        ns = self.config.namespace
        self.pub_cropped = rospy.Publisher(f"{ns}/image/cropped", Image, queue_size=1)
        self.pub_shear = rospy.Publisher(f"{ns}/image/shear", Image, queue_size=1)
        self.pub_divergence = rospy.Publisher(f"{ns}/image/divergence", Image, queue_size=1)
        self.pub_shear_diff = rospy.Publisher(f"{ns}/image/shear_diff", Image, queue_size=1)
        self.pub_modeled = rospy.Publisher(f"{ns}/image/modeled", Image, queue_size=1)
        self.pub_wrench = rospy.Publisher(f"{ns}/wrench/wrench_pred", WrenchStamped, queue_size=10)
        self.pub_force_norm = rospy.Publisher(f"{ns}/wrench/force_norm", Float64, queue_size=10)
        
        if HAS_TENSEGRITY_MSGS:
            self.pub_shear_vector = rospy.Publisher(f"{ns}/array/shear_vector", Float32MultiArrayStamped, queue_size=10)
        else:
            self.pub_shear_vector = rospy.Publisher(f"{ns}/array/shear_vector", Float32MultiArray, queue_size=10)
    
    def open_camera(self) -> bool:
        """Open the camera device."""
        self.cap = cv2.VideoCapture(self.config.camera_index)
        if not self.cap.isOpened():
            rospy.logerr(f"[{self.config.namespace}] Cannot open camera #{self.config.camera_index}")
            return False
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        rospy.loginfo(f"[{self.config.namespace}] Camera #{self.config.camera_index} opened")
        return True
    
    def close_camera(self):
        """Close the camera device."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
    
    def process_frame(self, frame: np.ndarray) -> Tuple[Dict[str, np.ndarray], torch.Tensor]:
        """Process a single frame through the shear pipeline."""
        # Convert and preprocess
        frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float().to(self.device)
        frame_tensor = square_center_crop(frame_tensor)
        frame_tensor = downsample(frame_tensor, GlobalConfig.PROCESSED_IMAGE_SIZE)
        
        self.shgen.update_time(self.frame_id * self.frame_period)
        self.shgen.update_tactile_image(frame_tensor)
        self.shgen.update_shear()
        
        # Extract shear components
        shear_field = self.shgen.get_shear_field()
        vf = get_channel(shear_field, [self.shgen.channels.index('u'), self.shgen.channels.index('v')])
        sf = get_channel(shear_field, self.shgen.channels.index('div'))
        diff_vf = get_channel(shear_field, [self.shgen.channels.index('du'), self.shgen.channels.index('dv')])
        
        # Build visualizations
        shear_img = cv_plot_vector_field(vf, ch_dim=0, image_size=GlobalConfig.DISPLAY_SIZE, color=GlobalConfig.SHEAR_COLOR)
        
        divergence_small = cv_plot_scalar_field(sf, max_magnitude=GlobalConfig.MAX_SHEAR_MAGNITUDE, colormap=cv2.COLORMAP_JET)
        divergence_img = cv2.resize(divergence_small, GlobalConfig.DISPLAY_SIZE, interpolation=cv2.INTER_NEAREST)
        
        shear_diff_img = cv_plot_vector_field(diff_vf, ch_dim=0, image_size=GlobalConfig.DISPLAY_SIZE, color=GlobalConfig.SHEAR_DIFF_COLOR)
        
        frame_cropped = center_crop(frame)
        frame_cropped = cv2.resize(frame_cropped, GlobalConfig.DISPLAY_SIZE, interpolation=cv2.INTER_LINEAR)
        
        # Contact modeling
        sf_numpy = sf.cpu().numpy()
        clipped = np.clip(sf_numpy, -GlobalConfig.MAX_SHEAR_MAGNITUDE, GlobalConfig.MAX_SHEAR_MAGNITUDE)
        normed = ((clipped + GlobalConfig.MAX_SHEAR_MAGNITUDE) / (2 * GlobalConfig.MAX_SHEAR_MAGNITUDE) * 255).astype(np.float32)
        
        coords = self._detect_contacts(sf)
        modeled_field = self._generate_contact_model(normed, coords)
        
        modeled_small = cv_plot_scalar_field(modeled_field, max_magnitude=128, colormap=cv2.COLORMAP_JET)
        modeled_img = cv2.resize(modeled_small, GlobalConfig.DISPLAY_SIZE, interpolation=cv2.INTER_LINEAR)
        
        # Add contact markers
        self._add_contact_markers([frame_cropped, shear_img, divergence_img], coords)
        
        images = {
            'cropped': frame_cropped,
            'shear': shear_img,
            'divergence': divergence_img,
            'shear_diff': shear_diff_img,
            'modeled': modeled_img,
        }
        
        return images, vf
    
    def _detect_contacts(self, shear_field: torch.Tensor) -> List[Tuple[int, int]]:
        """Detect contact points from shear field."""
        sf = shear_field.cpu().numpy()
        clipped = np.clip(sf, -GlobalConfig.MAX_SHEAR_MAGNITUDE, GlobalConfig.MAX_SHEAR_MAGNITUDE)
        normed = ((clipped + GlobalConfig.MAX_SHEAR_MAGNITUDE) / (2 * GlobalConfig.MAX_SHEAR_MAGNITUDE) * 255).astype(np.float32)
        
        coords = peak_local_max(normed, min_distance=GlobalConfig.CONTACT_MIN_DISTANCE, threshold_abs=GlobalConfig.CONTACT_THRESHOLD)
        return [(int(y), int(x)) for y, x in coords]
    
    def _generate_contact_model(self, normed: np.ndarray, coords: List[Tuple[int, int]]) -> np.ndarray:
        """Generate Gaussian mixture model of detected contacts."""
        modeled = np.zeros_like(normed)
        half_window = GlobalConfig.CONTACT_WINDOW_SIZE // 2
        
        for y0, x0 in coords:
            ymin = max(0, y0 - half_window)
            ymax = min(normed.shape[0], y0 + half_window + 1)
            xmin = max(0, x0 - half_window)
            xmax = min(normed.shape[1], x0 + half_window + 1)
            
            window = normed[ymin:ymax, xmin:xmax]
            mu_win, cov_win = local_moments(window, ymin, xmin)
            amplitude = window.max() * GlobalConfig.GAUSSIAN_AMPLITUDE_FACTOR
            g = gaussian_2d(normed.shape, mean=mu_win, cov=cov_win, amplitude=amplitude)
            modeled += g
        
        return modeled
    
    def _add_contact_markers(self, images: List[np.ndarray], coords: List[Tuple[int, int]]):
        """Add contact point markers to images."""
        scale = GlobalConfig.DISPLAY_SIZE[0] / GlobalConfig.SHEAR_OUTPUT_SIZE[0]
        
        for y0, x0 in coords:
            mx, my = int(x0 * scale), int(y0 * scale)
            for img in images:
                cv2.drawMarker(img, (mx, my), color=GlobalConfig.CONTACT_COLOR,
                              markerType=cv2.MARKER_CROSS, markerSize=GlobalConfig.MARKER_SIZE,
                              thickness=GlobalConfig.MARKER_THICKNESS)
    
    def infer_wrench(self, vf: torch.Tensor) -> np.ndarray:
        """Run wrench inference on vector field."""
        if self.model is None:
            return np.zeros(6)
        
        x = vf.cpu().numpy().flatten()
        xt = torch.from_numpy(x).float().unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            y = self.model(xt).squeeze(0).cpu().numpy()
        
        if y.shape[0] < 6:
            y = np.pad(y, (0, 6 - y.shape[0]))
        
        # Zero out small forces
        if np.linalg.norm(y[:3]) < GlobalConfig.FORCE_THRESHOLD:
            y[:] = 0.0
        
        return y
    
    def publish(self, images: Dict[str, np.ndarray], vf: torch.Tensor, wrench: np.ndarray):
        """Publish all outputs to ROS."""
        now = rospy.Time.now()
        frame_id = self.config.frame_id
        
        # Publish images
        for name, img in images.items():
            msg = cv2_to_imgmsg(img, encoding='bgr8')
            msg.header.stamp = now
            msg.header.frame_id = frame_id
            getattr(self, f'pub_{name}').publish(msg)
        
        # Publish shear vector
        vf_numpy = vf.cpu().numpy().astype(np.float32)
        if HAS_TENSEGRITY_MSGS:
            vf_msg = Float32MultiArrayStamped()
            vf_msg.header.stamp = now
            vf_msg.header.frame_id = frame_id
        else:
            vf_msg = Float32MultiArray()
        
        vf_msg.layout.dim = [
            MultiArrayDimension(label="channels", size=vf_numpy.shape[0], stride=vf_numpy.size),
            MultiArrayDimension(label="height", size=vf_numpy.shape[1], stride=vf_numpy.shape[1] * vf_numpy.shape[2]),
            MultiArrayDimension(label="width", size=vf_numpy.shape[2], stride=vf_numpy.shape[2])
        ]
        vf_msg.data = vf_numpy.flatten().tolist()
        self.pub_shear_vector.publish(vf_msg)
        
        # Publish wrench
        ws = WrenchStamped()
        ws.header.stamp = now
        ws.header.frame_id = frame_id
        ws.wrench.force.x = -float(wrench[0])
        ws.wrench.force.y = -float(wrench[1])
        ws.wrench.force.z = -float(wrench[2])
        ws.wrench.torque.x = -float(wrench[3])
        ws.wrench.torque.y = -float(wrench[4])
        ws.wrench.torque.z = -float(wrench[5])
        self.pub_wrench.publish(ws)
        
        # Publish force magnitude (2-norm)
        force_norm = float(np.linalg.norm(wrench[:3]))
        self.pub_force_norm.publish(Float64(data=force_norm))
    
    def run_once(self) -> bool:
        """Run one iteration of the pipeline."""
        if self.cap is None:
            return False
        
        ret, frame = self.cap.read()
        if not ret:
            rospy.logwarn(f"[{self.config.namespace}] Failed to grab frame")
            return False
        
        images, vf = self.process_frame(frame)
        wrench = self.infer_wrench(vf)
        self.publish(images, vf, wrench)
        
        self.frame_id += 1
        return True
    
    def run(self):
        """Main loop for this pipeline."""
        if not self.open_camera():
            return
        
        self.running = True
        rate = rospy.Rate(GlobalConfig.FRAME_RATE)
        
        try:
            while self.running and not rospy.is_shutdown():
                if not self.run_once():
                    break
                rate.sleep()
        finally:
            self.close_camera()
    
    def stop(self):
        """Stop the pipeline."""
        self.running = False


# =============================================================================
# Multi-Camera Manager
# =============================================================================

class MultiCameraManager:
    """Manages multiple camera pipelines in parallel threads."""
    
    def __init__(self, camera_indices: List[int]):
        self.camera_indices = camera_indices
        self.pipelines: List[CameraPipeline] = []
        self.threads: List[threading.Thread] = []
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        rospy.loginfo(f"Using device: {self.device}")
        
        # Load model once, share across pipelines
        self.model = self._load_model()
        
        # Create pipelines
        for i, cam_idx in enumerate(camera_indices):
            endcap_id = GlobalConfig.CAMERA_TO_ENDCAP.get(cam_idx, i)
            config = CameraConfig(
                camera_index=cam_idx,
                namespace=f"endcap_{endcap_id}",
                frame_id=f"endcap_{endcap_id}_grasp_frame"
            )
            pipeline = CameraPipeline(config, self.device, self.model)
            self.pipelines.append(pipeline)
    
    def _load_model(self) -> Optional[ResNetRegressor]:
        """Load the wrench inference model."""
        ckpt_path = find_checkpoint()
        if ckpt_path is None:
            rospy.logwarn("No checkpoint found, wrench inference disabled")
            return None
        
        # Determine input dimension from shear output size
        in_dim = 2 * GlobalConfig.SHEAR_OUTPUT_SIZE[0] * GlobalConfig.SHEAR_OUTPUT_SIZE[1]
        model = ResNetRegressor(in_dim, 6).to(self.device)
        
        try:
            state = torch.load(ckpt_path, map_location=self.device)
            model.load_state_dict(state)
            model.eval()
            rospy.loginfo(f"Loaded model from: {ckpt_path}")
        except Exception as e:
            rospy.logerr(f"Failed to load model: {e}")
            return None
        
        return model
    
    def start(self):
        """Start all pipelines in separate threads."""
        for pipeline in self.pipelines:
            thread = threading.Thread(target=pipeline.run, daemon=True)
            thread.start()
            self.threads.append(thread)
            rospy.loginfo(f"Started pipeline: {pipeline.config.namespace}")
    
    def stop(self):
        """Stop all pipelines."""
        for pipeline in self.pipelines:
            pipeline.stop()
        
        for thread in self.threads:
            thread.join(timeout=2.0)
    
    def wait(self):
        """Wait for all pipelines to finish."""
        try:
            rospy.spin()
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()


# =============================================================================
# Main
# =============================================================================

def test_cameras(camera_indices: List[int], save_images: bool = False, warmup_frames: int = 30) -> Dict[int, bool]:
    """Test which cameras are available and optionally save sample images.
    
    Args:
        camera_indices: List of camera indices to test
        save_images: If True, save a sample image from each camera
        warmup_frames: Number of frames to discard for auto-exposure to stabilize
    """
    results = {}
    for idx in camera_indices:
        cap = cv2.VideoCapture(idx)
        results[idx] = cap.isOpened()
        if cap.isOpened():
            if save_images:
                # Discard frames to let auto-exposure stabilize
                endcap_id = GlobalConfig.CAMERA_TO_ENDCAP.get(idx, idx)
                print(f"  Camera {idx} (camera_{idx}-endcap {endcap_id}): warming up ({warmup_frames} frames)...", end=" ", flush=True)
                for _ in range(warmup_frames):
                    cap.read()
                # Now grab the stabilized frame
                ret, frame = cap.read()
                if ret and frame is not None:
                    filename = f"camera_{idx}-endcap_{endcap_id}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"saved {filename}")
                else:
                    print("failed to grab frame")
            cap.release()
    return results


def main():
    parser = argparse.ArgumentParser(description="Run 6-camera GelFoot pipelines")
    parser.add_argument("--cameras", type=str, default="0,2,4,6,8,10",
                        help="Comma-separated camera indices (default: 0,2,4,6,8,10)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Test camera availability without running pipelines")
    parser.add_argument("--save-img", action="store_true",
                        help="Save a sample image from each camera")
    args = parser.parse_args()
    
    camera_indices = [int(x.strip()) for x in args.cameras.split(",")]
    
    # Test cameras first
    print(f"Testing cameras: {camera_indices}")
    test_results = test_cameras(camera_indices, save_images=args.save_img)
    
    for idx, available in test_results.items():
        endcap_id = GlobalConfig.CAMERA_TO_ENDCAP.get(idx, idx)
        status = "OK" if available else "NOT FOUND"
        print(f"  Camera {idx} -> endcap {endcap_id}: {status}")
    
    if args.dry_run:
        available_count = sum(test_results.values())
        print(f"\nAvailable: {available_count}/{len(camera_indices)} cameras")
        return
    
    # Filter to only available cameras
    available_cameras = [idx for idx, ok in test_results.items() if ok]
    if not available_cameras:
        print("ERROR: No cameras available!")
        sys.exit(1)
    
    if len(available_cameras) < len(camera_indices):
        print(f"\nWARNING: Only {len(available_cameras)}/{len(camera_indices)} cameras available")
        print(f"Proceeding with cameras: {available_cameras}")
    
    # Initialize ROS
    rospy.init_node("gelfoot_multi_camera", anonymous=True)
    
    # Start pipelines
    manager = MultiCameraManager(available_cameras)
    manager.start()
    
    print(f"\n{'='*50}")
    print(f"Running {len(available_cameras)} camera pipelines")
    print(f"Press Ctrl+C to stop")
    print(f"{'='*50}\n")
    
    manager.wait()
    print("\nShutdown complete.")


if __name__ == "__main__":
    main()

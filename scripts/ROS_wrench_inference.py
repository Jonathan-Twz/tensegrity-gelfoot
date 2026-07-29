#!/usr/bin/env python3
"""
Single-file ROS node for GelFoot wrench inference.

Subscribes:
  - /gelslim/array/shear_vector (tensegrity_msgs/Float32MultiArrayStamped)

Loads model from:
  - checkpoints/wrench_regression_model.pth (relative to this script),
    or the first .pth file found under checkpoints/

Publishes:
  - /gelslim/wrench/wrench_pred (geometry_msgs/WrenchStamped)
"""

import os
import glob
from typing import Optional

import numpy as np
import rospy
import torch
import torch.nn as nn
from geometry_msgs.msg import WrenchStamped
from std_msgs.msg import Header
from tensegrity_msgs.msg import Float32MultiArrayStamped


# -----------------------------
# Minimal model (inline)
# -----------------------------
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


# -----------------------------
# Node implementation
# -----------------------------
class ROSWrenchInference:
    def __init__(self):
        # Topics (relative for namespace support)
        self.topic_in = rospy.get_param("~topic_in", "gelslim/array/shear_vector")
        self.topic_out = rospy.get_param("~topic_out", "gelslim/wrench/wrench_pred")
        self.frame_id = rospy.get_param("~frame_id", "grasp_frame")

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rospy.loginfo(f"ROSWrenchInference using device: {self.device}")

        # Model will be created lazily on first message when input_dim is known
        self.model: Optional[ResNetRegressor] = None
        self.model_loaded = False
        self.out_dim = 6

        # Resolve checkpoint
        self.ckpt_path = self._find_checkpoint()
        if self.ckpt_path is None:
            rospy.logwarn("No checkpoint found under checkpoints/. Node will wait but cannot infer until a checkpoint is provided.")

        # ROS I/O
        self.sub = rospy.Subscriber(self.topic_in, Float32MultiArrayStamped, self.cb_vf, queue_size=10)
        self.pub = rospy.Publisher(self.topic_out, WrenchStamped, queue_size=10)

    def _find_checkpoint(self) -> Optional[str]:
        base = os.path.dirname(os.path.abspath(__file__))
        ckpt_dir = os.path.join(base, "checkpoints")
        candidate = os.path.join(ckpt_dir, "wrench_regression_model.pth")
        if os.path.isfile(candidate):
            return candidate
        # fallback: first .pth in dir
        pths = sorted(glob.glob(os.path.join(ckpt_dir, "*.pth")))
        if pths:
            return pths[0]
        return None

    def _ensure_model(self, in_dim: int):
        if self.model is not None:
            return
        if in_dim <= 0:
            raise ValueError("Invalid input dimension")
        self.model = ResNetRegressor(in_dim, self.out_dim).to(self.device)
        self.model.eval()
        if self.ckpt_path and os.path.isfile(self.ckpt_path):
            try:
                state = torch.load(self.ckpt_path, map_location=self.device)
                self.model.load_state_dict(state)
                self.model_loaded = True
                rospy.loginfo(f"Loaded model weights from: {self.ckpt_path}")
            except Exception as e:
                rospy.logerr(f"Failed to load checkpoint: {self.ckpt_path}. Error: {e}")
        else:
            rospy.logwarn("Checkpoint path not available; running with uninitialized weights")

    def cb_vf(self, msg: Float32MultiArrayStamped):
        # Validate layout [C,H,W]
        dims = msg.layout.dim
        if len(dims) != 3:
            rospy.logwarn_throttle(5.0, f"Expected 3 dims [C,H,W], got {len(dims)}; skipping")
            return

        C, H, W = dims[0].size, dims[1].size, dims[2].size
        expected = C * H * W
        data = np.asarray(msg.data, dtype=np.float32)
        if data.size != expected:
            rospy.logwarn_throttle(5.0, f"Data size mismatch: layout implies {expected}, data has {data.size}")
            return

        # Lazily build model
        try:
            self._ensure_model(expected)
        except Exception as e:
            rospy.logerr(f"Cannot initialize model: {e}")
            return

        # Flatten input and run inference
        x = data.reshape(-1)  # (C*H*W,)
        xt = torch.from_numpy(x).float().unsqueeze(0).to(self.device)  # (1, D)
        with torch.no_grad():
            y = self.model(xt).squeeze(0).cpu().numpy()

        # Build and publish WrenchStamped
        ws = WrenchStamped()
        ws.header = Header()
        ws.header.stamp = msg.header.stamp if msg.header.stamp else rospy.Time.now()
        ws.header.frame_id = self.frame_id

        if y.shape[0] < 6:
            y = np.pad(y, (0, 6 - y.shape[0]))

        # NOTE: invert to sensor frame
        ws.wrench.force.x = -float(y[0])
        ws.wrench.force.y = -float(y[1])
        ws.wrench.force.z = -float(y[2])
        ws.wrench.torque.x = -float(y[3])
        ws.wrench.torque.y = -float(y[4])
        ws.wrench.torque.z = -float(y[5])

        # NOTE: Zero out small forces to reduce noise
        force_norm = np.linalg.norm(y[0:3])
        if force_norm < 2.0:
            ws.wrench.force.x = 0.0
            ws.wrench.force.y = 0.0
            ws.wrench.force.z = 0.0
            ws.wrench.torque.x = 0.0
            ws.wrench.torque.y = 0.0
            ws.wrench.torque.z = 0.0

        self.pub.publish(ws)
        rospy.loginfo_throttle(period=2, msg=f"Published wrench: force=({y[0]:.2f}, {y[1]:.2f}, {y[2]:.2f}), torque=({y[3]:.2f}, {y[4]:.2f}, {y[5]:.2f})")


def main():
    rospy.init_node("ROS_wrench_inference", anonymous=False)
    _ = ROSWrenchInference()
    rospy.loginfo("ROS_wrench_inference node started. Waiting for /gelslim/array/shear_vector...")
    rospy.spin()


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass

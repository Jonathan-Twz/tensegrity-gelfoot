#!/usr/bin/env python3
"""Experiment script for BubbleMed shear data collection."""

import math
import json
from pathlib import Path
from typing import List

import rospy
from bubble_utils.bubble_med.bubble_med import BubbleMed
from geometry_msgs.msg import WrenchStamped
from std_msgs.msg import Bool
from victor_hardware_interface_msgs.msg import ControlMode

from mmint_utils.gamma_helpers.gamma_helpers import zero_ati_gamma


DISTAL_FRAME = 'grasp_frame'
REF_FRAME = 'med_base'
WRENCH_TOPIC = '/wrench_grasp_frame'
FLAG_TOPIC = '/dataset/is_recording'

# Default joint configurations for BubbleMed (seven joints each)

DEFAULT_HOME_CONFIGS = [
    [1.8259369995544348, 0.004541481976706276, -0.08067109090373113, -1.9625600397508363, -1.4462202129753743, 1.0203964700253507, -2],
    [1.8038446202464444, -0.018746383728050226, -0.0805348302128679, -1.927480835212026, -1.2514671626664562, 0.9421893956896767, -2],
    [1.7659023245882868, -0.045450352707137945, -0.08053500998178086, -1.8762760173147477, -0.9772513189599817, 0.8701722334297961, -2]
]

DEFAULT_PROGRESS_PATH = Path.home() / '.cache' / 'bubblemed_progress.json'

print(f"Default progress path: {DEFAULT_PROGRESS_PATH}")
# import pdb; pdb.set_trace()

class ExperimentRunner:
    """Run layered joint / Cartesian exploration while monitoring wrench."""

    def __init__(self) -> None:
        rospy.init_node('bubblemed_dataset_runner', anonymous=False)

        self.robot = BubbleMed()
        self.flag_pub = rospy.Publisher(FLAG_TOPIC, Bool, queue_size=10)

        # Parameters for the experiment
        self.target_force = rospy.get_param('~target_force', 20.0)
        self.joint_velocity = rospy.get_param('~joint_velocity', 0.09)  # rad/s
        self.descent_step = rospy.get_param('~descent_step', 0.001)  # meters (1 mm)
        self.max_descent_steps = rospy.get_param('~max_descent_steps', 120)
        self.circle_radius = rospy.get_param('~circle_radius', 0.015)  # meters (20 mm)
        self.circle_steps = rospy.get_param('~circle_steps', 13)
        self.joint7_start = rospy.get_param('~joint7_start', -2)
        self.joint7_stop = rospy.get_param('~joint7_stop', 0)
        self.joint7_step = rospy.get_param('~joint7_step', 1.01)
        self.wrench_timeout = rospy.get_param('~wrench_timeout', 1.0)

        home_configs_param = rospy.get_param('~home_configs', DEFAULT_HOME_CONFIGS)
        self.home_configs = [list(cfg) for cfg in home_configs_param]
        if not self.home_configs:
            raise ValueError('home_configs parameter must contain at least one configuration')
        for cfg in self.home_configs:
            if len(cfg) != 7:
                raise ValueError(f'Each home configuration must have 7 joints, got {len(cfg)}')

        progress_param = rospy.get_param('~progress_path', str(DEFAULT_PROGRESS_PATH))
        self.progress_path = Path(progress_param).expanduser().resolve()
        self.progress = self._load_progress()
        self.joint7_targets = self._compute_joint_targets()

        self._configure_robot()

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------
    def _configure_robot(self) -> None:
        rospy.loginfo('Configuring robot control mode to JOINT_POSITION (vel=%.3f)', self.joint_velocity)
        self.robot.set_control_mode(ControlMode.JOINT_POSITION, vel=self.joint_velocity)
        self.robot.set_joint_position_control(vel=self.joint_velocity)

    def _compute_joint_targets(self) -> List[float]:
        if self.joint7_step == 0:
            raise ValueError('joint7_step must be non-zero')
        span = self.joint7_stop - self.joint7_start
        count = int(round(span / self.joint7_step)) if span != 0 else 0
        # Ensure we include the final value even if rounding would drop it
        targets = [round(self.joint7_start + i * self.joint7_step, 6) for i in range(count + 1)]
        if targets and targets[-1] < self.joint7_stop - 1e-6:
            targets.append(round(self.joint7_stop, 6))
        if not targets:
            targets = [round(self.joint7_start, 6)]
        return targets

    def _load_progress(self) -> dict:
        if not self.progress_path.exists():
            return {'home_index': 0, 'joint_index': 0}
        try:
            with self.progress_path.open('r', encoding='utf-8') as handle:
                data = json.load(handle)
            home_idx = int(data.get('home_index', 0))
            joint_idx = int(data.get('joint_index', 0))
            return {'home_index': max(home_idx, 0), 'joint_index': max(joint_idx, 0)}
        except Exception as exc:
            rospy.logwarn('Failed to load progress from %s: %s', self.progress_path, exc)
            return {'home_index': 0, 'joint_index': 0}

    def _save_progress(self, home_index: int, joint_index: int) -> None:
        try:
            self.progress_path.parent.mkdir(parents=True, exist_ok=True)
            with self.progress_path.open('w', encoding='utf-8') as handle:
                json.dump({'home_index': home_index, 'joint_index': joint_index}, handle)
            self.progress['home_index'] = home_index
            self.progress['joint_index'] = joint_index
        except Exception as exc:
            rospy.logwarn('Failed to save progress to %s: %s', self.progress_path, exc)

    def _clear_progress(self) -> None:
        self.progress = {'home_index': 0, 'joint_index': 0}
        if self.progress_path.exists():
            try:
                self.progress_path.unlink()
            except Exception as exc:
                rospy.logwarn('Failed to remove progress file %s: %s', self.progress_path, exc)

    def _joint_config_with_j7(self, base_joints: List[float], joint7: float) -> List[float]:
        joints = list(base_joints)
        joints[6] = joint7
        return joints

    def _plan_to_joints(self, target_joints: List[float]) -> None:
        rospy.loginfo('Planning to joints: %s', ['{:.3f}'.format(v) for v in target_joints])
        self.flag_pub.publish(Bool(data=True))
        result = self.robot.plan_to_joint_config(self.robot.arm_group, target_joints)
        if hasattr(result, 'success') and not result.success:
            rospy.logwarn('Joint plan reported failure; attempting direct set')
            self.robot.set_joint_positions(target_joints, vel=self.joint_velocity)
        rospy.sleep(0.5)

    # ------------------------------------------------------------------
    # Wrench utilities
    # ------------------------------------------------------------------
    def _read_wrench_norm(self):
        try:
            msg = rospy.wait_for_message(WRENCH_TOPIC, WrenchStamped, timeout=self.wrench_timeout)
        except rospy.ROSException:
            rospy.logwarn_throttle(5.0, 'Timed out waiting for wrench message on %s', WRENCH_TOPIC)
            return None
        force = msg.wrench.force
        return math.sqrt(force.x ** 2 + force.y ** 2 + force.z ** 2)

    # ------------------------------------------------------------------
    # Motion primitives
    # ------------------------------------------------------------------
    def _set_pose(self, pose: List[float]) -> None:
        self.flag_pub.publish(Bool(data=True))
        self.robot.set_pose(pose=pose, frame_id=DISTAL_FRAME, ref_frame=REF_FRAME, position_tol=0.0005)

    def _execute_xy_circle(self, center_pose: List[float]) -> None:
        cx, cy = center_pose[0], center_pose[1]
        base_pose = list(center_pose)
        for step_idx in range(self.circle_steps):
            angle = -2.0 * math.pi * step_idx / self.circle_steps
            target = list(base_pose)
            target[0] = cx + self.circle_radius * math.cos(angle)
            target[1] = cy + self.circle_radius * math.sin(angle)
            self._set_pose(target)
            rospy.sleep(0.4)
        # Return to center to avoid drift before next descent step
        self._set_pose(base_pose)

    def _perform_descent(self, start_pose: List[float]) -> None:
        pose = list(start_pose)
        original_xy = pose[0], pose[1]
        start_z = pose[2]
        contact_reached = False

        for step in range(self.max_descent_steps):
            if rospy.is_shutdown():
                break

            norm_force = self._read_wrench_norm()
            if norm_force is None:
                continue

            rospy.loginfo('Step %d | force = %.3f N', step, norm_force)

            if norm_force >= self.target_force:
                contact_reached = True
                rospy.loginfo('Target force %.2f N reached; executing circle', self.target_force)
                self.flag_pub.publish(Bool(data=True))
                self._execute_xy_circle(pose)
                break

            pose[0], pose[1] = original_xy
            pose[2] -= self.descent_step
            self._set_pose(pose)

        if contact_reached:
            self.flag_pub.publish(Bool(data=False))
        pose[0], pose[1], pose[2] = original_xy[0], original_xy[1], start_z
        self._set_pose(pose)

    # ------------------------------------------------------------------
    # Main routine
    # ------------------------------------------------------------------
    def run(self) -> None:
        total_homes = len(self.home_configs)
        targets = self.joint7_targets

        progress_joint = int(self.progress.get('joint_index', 0))
        progress_home = int(self.progress.get('home_index', 0))

        if progress_joint >= len(targets):
            rospy.loginfo('Existing progress indicates previous run completed; restarting from beginning')
            progress_joint = 0
            progress_home = 0

        rospy.loginfo('Starting demo with progress joint_index=%d, home_index=%d', progress_joint, progress_home)

        for joint_idx, joint7 in enumerate(targets):
            if rospy.is_shutdown():
                self._save_progress(progress_home, joint_idx)
                return
            if joint_idx < progress_joint:
                continue

            rospy.loginfo('=== Joint 7 target %.2f rad (%d/%d) ===', joint7, joint_idx + 1, len(targets))
            start_home_idx = progress_home if joint_idx == progress_joint else 0

            for home_idx, home_cfg in enumerate(self.home_configs):
                if rospy.is_shutdown():
                    self._save_progress(home_idx, joint_idx)
                    return
                if home_idx < start_home_idx:
                    continue

                rospy.loginfo('--- Home configuration %d/%d ---', home_idx + 1, total_homes)
                target_joints = self._joint_config_with_j7(home_cfg, joint7)
                self._plan_to_joints(target_joints)
                self._save_progress(home_idx, joint_idx)

                try:
                    zero_ati_gamma()
                    current_pose = list(self.robot.get_current_pose())
                    rospy.loginfo('Starting descent from pose: %s', [f"{v:.4f}" for v in current_pose])
                    self._perform_descent(current_pose)
                except Exception:
                    self._save_progress(home_idx, joint_idx)
                    raise
                else:
                    self._plan_to_joints(self._joint_config_with_j7(home_cfg, joint7))
                    self._save_progress(home_idx + 1, joint_idx)

            progress_home = 0
            progress_joint = joint_idx + 1
            self._save_progress(progress_home, progress_joint)

        self._clear_progress()
        rospy.loginfo('Demo finished; returning to initial home configuration')
        self._plan_to_joints(self.home_configs[0])


def main() -> None:
    runner = ExperimentRunner()
    runner.run()


if __name__ == '__main__':
    main()

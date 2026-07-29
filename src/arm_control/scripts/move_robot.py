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
# DEFAULT_HOME_CONFIGS = [
#     [
#         -1.5571975271691387,
#         -0.21685263217621578,
#         -0.5989819870530589,
#         1.6159687053844756,
#         1.5529173013465019,
#         -1.065497063113712,
#         -1.0476360740068908,
#     ],
#     [-1.5617010791051253, -0.2145817414167962, -0.5929887960550799, 1.5821038624328763, 1.4428036543446412, -1.0695462930826842, 0.9246437114387023],
#     [-1.5616956861468199, -0.21688564867875884, -0.5929897547862546, 1.5657367668462654, 1.3835722458737063, -1.07879847423381, 0.9905384927748728],
#     [-1.5621486898448336, -0.2196442974412547, -0.5929903539796035, 1.5259788610177938, 1.286352474411689, -1.1054214897152854, 1.106175237641695],
#     [-1.5634553299348404, -0.22740210565204114, -0.5929914325930813, 1.4903618044720486, 1.1954176138638826, -1.1428343224006416, 1.212720270092229],
#     [-1.5686417429224175, -0.24568044552541485, -0.5929929305219117, 1.4113582586224551, 1.0569051889096512, -1.2360795017646842, 1.3885062591187036],
#     [-1.5739339167540132, -0.26275526950571554, -0.59299203178643, 1.3550960976719446, 0.9810809978973266, -1.3131164963524302, 1.4913758497122698],
#     [-1.5821585704327785, -0.29017637453041056, -0.5929929305219117, 1.286783499069349, 0.9012751680638053, -1.4212823352487305, 1.6039097588159548],
#     [-1.5797110920592974, -0.31570109576517585, -0.5929931102908247, 1.2038665044124344, 0.7766396125187502, -1.5183437712700378, 1.6973224776799674]
# ]
DEFAULT_HOME_CONFIGS = [
    [1.8259369995544348, 0.004541481976706276, -0.08067109090373113, -1.9625600397508363, -1.4462202129753743, 1.0203964700253507, -0.3284508729949839],
    [1.8038446202464444, -0.018746383728050226, -0.0805348302128679, -1.927480835212026, -1.2514671626664562, 0.9421893956896767, -0.3837942632838965],
    [1.7659023245882868, -0.045450352707137945, -0.08053500998178086, -1.8762760173147477, -0.9772513189599817, 0.8701722334297961, -0.49433423712793645]
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
        self.min_force = rospy.get_param('~min_force', 0.9)
        self.max_force = rospy.get_param('~max_force', 32.0)
        self.joint_velocity = rospy.get_param('~joint_velocity', 0.02)  # rad/s
        self.descent_step = rospy.get_param('~descent_step', 0.001)  # meters (1 mm)
        self.max_descent_steps = rospy.get_param('~max_descent_steps', 120)
        self.circle_radius = rospy.get_param('~circle_radius', 0.015)  # meters (20 mm)
        self.circle_steps = rospy.get_param('~circle_steps', 13)
        self.joint7_start = rospy.get_param('~joint7_start', -2.9)
        self.joint7_stop = rospy.get_param('~joint7_stop', 2.9)
        self.joint7_step = rospy.get_param('~joint7_step', 0.2)
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
        # Return to center to avoid drift before next descent step
        self._set_pose(base_pose)

    def _perform_descent(self, start_pose: List[float]) -> None:
        pose = list(start_pose)
        original_xy = pose[0], pose[1]
        start_z = pose[2]
        contact_started = False

        for step in range(self.max_descent_steps):
            if rospy.is_shutdown():
                break

            norm_force = self._read_wrench_norm()
            if norm_force is None:
                continue

            rospy.loginfo('Step %d | force = %.3f N', step, norm_force)

            if norm_force >= self.max_force:
                rospy.loginfo('Max force %.2f reached (threshold %.2f); stopping descent', norm_force, self.max_force)
                break

            if not contact_started and norm_force >= self.min_force:
                contact_started = True
                rospy.loginfo('Min force %.2f reached; starting dataset capture', self.min_force)
                self.flag_pub.publish(Bool(data=True))

            # XY exploration once contact established
            if contact_started:
                self._execute_xy_circle(pose)

            # Move down 1 mm for the next iteration
            pose[0], pose[1] = original_xy
            pose[2] -= self.descent_step
            self._set_pose(pose)

        # Reset capture flag and retreat to starting height
        if contact_started:
            self.flag_pub.publish(Bool(data=False))
        pose[0], pose[1], pose[2] = original_xy[0], original_xy[1], start_z
        self._set_pose(pose)

    # ------------------------------------------------------------------
    # Main routine
    # ------------------------------------------------------------------
    def run(self) -> None:
        total_configs = len(self.home_configs)
        targets = self.joint7_targets

        progress_home = int(self.progress.get('home_index', 0))
        progress_joint = int(self.progress.get('joint_index', 0))

        if progress_home >= total_configs:
            rospy.loginfo('Existing progress indicates previous run completed; restarting from beginning')
            progress_home = 0
            progress_joint = 0
        elif progress_joint >= len(targets):
            progress_home = min(progress_home + 1, total_configs - 1)
            progress_joint = 0

        rospy.loginfo('Starting experiment with progress home_index=%d, joint_index=%d', progress_home, progress_joint)

        for cfg_idx, home_cfg in enumerate(self.home_configs, start=1):
            home_index = cfg_idx - 1
            if rospy.is_shutdown():
                self._save_progress(home_index, progress_joint)
                return
            if home_index < progress_home:
                continue

            rospy.loginfo('=== Home configuration %d/%d ===', cfg_idx, total_configs)

            joint_start_idx = progress_joint if home_index == progress_home else 0
            if joint_start_idx >= len(targets):
                rospy.loginfo('Home %d already completed; skipping to next configuration', cfg_idx)
                self._save_progress(home_index + 1, 0)
                continue

            start_joint_value = targets[joint_start_idx]
            start_config = self._joint_config_with_j7(home_cfg, start_joint_value)
            self._plan_to_joints(start_config)
            self._save_progress(home_index, joint_start_idx)

            for joint_idx, joint7 in enumerate(targets):
                if home_index == progress_home and joint_idx < joint_start_idx:
                    continue
                if rospy.is_shutdown():
                    self._save_progress(home_index, joint_idx)
                    return

                rospy.loginfo('--- Home %d | Joint 7 target: %.2f rad ---', cfg_idx, joint7)
                target_joints = self._joint_config_with_j7(home_cfg, joint7)
                self._save_progress(home_index, joint_idx)
                self._plan_to_joints(target_joints)

                try:
                    zero_ati_gamma()
                    current_pose = list(self.robot.get_current_pose())
                    rospy.loginfo('Starting descent from pose: %s', [f"{v:.4f}" for v in current_pose])
                    self._perform_descent(current_pose)
                except Exception:
                    self._save_progress(home_index, joint_idx)
                    raise
                else:
                    self._save_progress(home_index, joint_idx + 1)

            # Return to the base home configuration before moving to the next one
            self._plan_to_joints(home_cfg)
            self._save_progress(home_index + 1, 0)
            progress_home = home_index + 1
            progress_joint = 0

        self._clear_progress()
        rospy.loginfo('Experiment finished; returning to initial home configuration')
        self._plan_to_joints(self.home_configs[0])


def main() -> None:
    runner = ExperimentRunner()
    runner.run()


if __name__ == '__main__':
    main()

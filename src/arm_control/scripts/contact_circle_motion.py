#!/usr/bin/env python3
"""Contact routine that descends onto the FT sensor, executes clockwise and counter-clockwise circles, and returns home."""

import math
from typing import List, Optional

import rospy
from bubble_utils.bubble_med.bubble_med import BubbleMed
from geometry_msgs.msg import WrenchStamped
from victor_hardware_interface_msgs.msg import ControlMode

from mmint_utils.gamma_helpers.gamma_helpers import zero_ati_gamma


DISTAL_FRAME = 'grasp_frame'
REF_FRAME = 'med_base'
WRENCH_TOPIC = '/wrench_grasp_frame'

DEFAULT_HOME_JOINTS = [
    # 1.4624230408998644, -0.1607837169166461, -0.08071273609592988, -1.9450170519244732, 0.1923096106491429, 0.41265760990041994, -2.3344805127742365
    # 1.4718539667829238, -0.12067892766292218, -0.08074497363621116, -1.8527523019956766, 0.061695940260187467, 0.7036604764805493, -2.227980600902774
    1.8259369995544348, 0.004541481976706276, -0.08067109090373113, -1.9625600397508363, -1.4462202129753743, 1.0203964700253507, -0.3284508729949839
]


class ContactCircleMotion:
    """Move BubbleMed onto the FT sensor, draw circles, then return home."""

    def __init__(self) -> None:
        rospy.init_node('contact_circle_motion', anonymous=False)

        self.robot = BubbleMed()
        self.force_threshold = rospy.get_param('~force_threshold', 20.0)
        self.max_force = rospy.get_param('~max_force', 35.0)
        self.descent_step = rospy.get_param('~descent_step', 0.0005)  # meters
        self.max_descent_steps = rospy.get_param('~max_descent_steps', 200)
        self.circle_radius = rospy.get_param('~circle_radius', 0.015)  # meters
        self.circle_steps = rospy.get_param('~circle_steps', 36)
        self.circle_pause = rospy.get_param('~circle_pause', 1.0)
        self.wrench_timeout = rospy.get_param('~wrench_timeout', 1.0)

        home_param = rospy.get_param('~home_joints', DEFAULT_HOME_JOINTS)
        if len(home_param) != 7:
            raise ValueError('home_joints must contain 7 joint values')
        self.home_joints = list(home_param)

        self._configure_robot()

    def _configure_robot(self) -> None:
        rospy.loginfo('Setting control mode to JOINT_POSITION for contact circle routine')
        self.robot.set_control_mode(ControlMode.JOINT_POSITION, vel=0.02)
        self.robot.set_joint_position_control(vel=0.02)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def _move_to_home(self) -> None:
        rospy.loginfo('Moving to home joint configuration')
        result = self.robot.plan_to_joint_config(self.robot.arm_group, self.home_joints)
        if hasattr(result, 'success') and not result.success:
            rospy.logwarn('Planning to home failed, attempting direct joint command')
            self.robot.set_joint_positions(self.home_joints, vel=0.02)
        rospy.sleep(0.5)

    def _wait_for_wrench_norm(self) -> Optional[float]:
        try:
            msg = rospy.wait_for_message(WRENCH_TOPIC, WrenchStamped, timeout=self.wrench_timeout)
        except rospy.ROSException:
            rospy.logwarn_throttle(5.0, 'Timed out waiting for wrench on %s', WRENCH_TOPIC)
            return None
        force = msg.wrench.force
        return math.sqrt(force.x ** 2 + force.y ** 2 + force.z ** 2)

    def _set_pose(self, pose: List[float]) -> None:
        self.robot.set_pose(pose=pose, frame_id=DISTAL_FRAME, ref_frame=REF_FRAME, position_tol=0.001)

    # ------------------------------------------------------------------
    # Contact and motion primitives
    # ------------------------------------------------------------------
    def _descend_until_force(self, start_pose: List[float]) -> List[float]:
        rospy.loginfo('Descending until force reaches %.2f N', self.force_threshold)
        pose = list(start_pose)
        zero_ati_gamma()
        rospy.sleep(0.2)

        for step in range(self.max_descent_steps):
            if rospy.is_shutdown():
                break

            norm_force = self._wait_for_wrench_norm()
            if norm_force is None:
                continue

            rospy.loginfo('Descent step %d | force %.3f N', step, norm_force)
            if norm_force >= self.force_threshold:
                rospy.loginfo('Force threshold met; stopping descent')
                return pose
            if norm_force >= self.max_force:
                rospy.logwarn('Force %.2f exceeds max %.2f; aborting descent', norm_force, self.max_force)
                return pose

            pose[2] -= self.descent_step
            self._set_pose(pose)

        rospy.logwarn('Max descent steps reached without hitting force threshold')
        return pose

    def _draw_circle(self, center_pose: List[float], direction: int) -> None:
        if direction not in (-1, 1):
            raise ValueError('direction must be -1 (clockwise) or 1 (counter-clockwise)')

        base_pose = list(center_pose)
        cx, cy = base_pose[0], base_pose[1]
        rospy.loginfo('Drawing circle radius %.3f m | direction %s', self.circle_radius, 'CW' if direction == -1 else 'CCW')

        for step_idx in range(self.circle_steps):
            angle = direction * 2.0 * math.pi * (step_idx / float(self.circle_steps))
            target = list(base_pose)
            target[0] = cx + self.circle_radius * math.cos(angle)
            target[1] = cy + self.circle_radius * math.sin(angle)
            self._set_pose(target)

        self._set_pose(base_pose)

    # ------------------------------------------------------------------
    # Execution flow
    # ------------------------------------------------------------------
    def run(self) -> None:
        self._move_to_home()

        rospy.loginfo('Capturing starting pose')
        start_pose = list(self.robot.get_current_pose())
        contact_pose = self._descend_until_force(start_pose)

        rospy.sleep(self.circle_pause)
        self._draw_circle(contact_pose, direction=-1)  # Clockwise
        rospy.sleep(self.circle_pause)
        self._draw_circle(contact_pose, direction=1)   # Counter-clockwise

        rospy.loginfo('Returning to home pose')
        self._move_to_home()
        rospy.loginfo('Contact circle routine complete')


def main() -> None:
    motion = ContactCircleMotion()
    motion.run()


if __name__ == '__main__':
    main()

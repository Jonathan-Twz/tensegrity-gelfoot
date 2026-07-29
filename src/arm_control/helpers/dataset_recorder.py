#!/usr/bin/env python3
"""Record a single wrench/shear sample for each /dataset/is_recording trigger."""

import csv
import json
import threading
from pathlib import Path
from typing import List

import numpy as np
import rospy
from geometry_msgs.msg import WrenchStamped
from std_msgs.msg import Bool
from tensegrity_msgs.msg import Float32MultiArrayStamped

import message_filters


class SingleShotRecorder:
    """Capture one synchronized sample for every rising edge on the trigger topic."""

    def __init__(self):
        # Parameters
        self.wrench_topic = rospy.get_param('~wrench_topic', '/wrench_grasp_frame')
        self.shear_topic = rospy.get_param('~shear_topic', '/gelslim/array/shear_vector')
        self.trigger_topic = rospy.get_param('~trigger_topic', '/dataset/is_recording')
        default_dir = Path.home() / 'gelfoot_dataset'
        self.output_dir = Path(rospy.get_param('~output_dir', str(default_dir))).expanduser().resolve()
        self.shear_dir = self.output_dir / 'shear_fields'
        self.index_file = self.output_dir / 'index.csv'
        self.sync_queue = rospy.get_param('~sync_queue', 10)
        self.sync_slop = rospy.get_param('~sync_slop', 0.05)

        # Internal state
        self._lock = threading.Lock()
        self._pending_captures = 0
        self._trigger_state = False
        self._next_index = 0

        self._prepare_storage()
        self._restore_index()

        # Subscribers
        self._trigger_sub = rospy.Subscriber(self.trigger_topic, Bool, self._trigger_cb, queue_size=10)
        wrench_sub = message_filters.Subscriber(self.wrench_topic, WrenchStamped)
        shear_sub = message_filters.Subscriber(self.shear_topic, Float32MultiArrayStamped)
        self._sync = message_filters.ApproximateTimeSynchronizer(
            [wrench_sub, shear_sub],
            queue_size=self.sync_queue,
            slop=self.sync_slop,
            allow_headerless=True,
        )
        self._sync.registerCallback(self._sync_cb)

        rospy.loginfo('SingleShotRecorder ready. Writing samples to %s', self.output_dir)

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------
    def _prepare_storage(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.shear_dir.mkdir(parents=True, exist_ok=True)
        if not self.index_file.exists():
            with self.index_file.open('w', newline='') as handle:
                writer = csv.writer(handle)
                writer.writerow(['idx', 'timestamp_nsec', 'wrench', 'shear_path'])

    def _restore_index(self) -> None:
        if not self.index_file.exists():
            return
        with self.index_file.open('r', newline='') as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                try:
                    idx = int(row['idx'])
                except (KeyError, TypeError, ValueError):
                    continue
                self._next_index = max(self._next_index, idx + 1)

    # ------------------------------------------------------------------
    # ROS Callbacks
    # ------------------------------------------------------------------
    def _trigger_cb(self, msg: Bool) -> None:
        if msg.data:
            with self._lock:
                self._pending_captures += 1
                pending = self._pending_captures
                self._trigger_state = True
            rospy.loginfo('Capture trigger received (%d pending)', pending)
        else:
            with self._lock:
                was_pending = self._pending_captures > 0
                self._pending_captures = 0
                was_active = self._trigger_state or was_pending
                self._trigger_state = False
            if was_active:
                rospy.loginfo('Capture trigger reset')

    def _sync_cb(self, wrench_msg: WrenchStamped, shear_msg: Float32MultiArrayStamped) -> None:
        with self._lock:
            if self._pending_captures <= 0:
                return
            self._pending_captures -= 1
            remaining = self._pending_captures
        try:
            idx = self._store_sample(wrench_msg, shear_msg)
            rospy.loginfo('Recorded sample %06d (%d pending)', idx, remaining)
        except Exception as exc:  # noqa: BLE001
            rospy.logerr('Failed to record sample: %s', exc)

    # ------------------------------------------------------------------
    # Storage helpers
    # ------------------------------------------------------------------
    def _store_sample(self, wrench_msg: WrenchStamped, shear_msg: Float32MultiArrayStamped) -> int:
        idx = self._reserve_index()
        timestamp = self._choose_timestamp(wrench_msg, shear_msg)
        wrench_values = self._wrench_to_list(wrench_msg)

        shear_path = self.shear_dir / f'{idx:06d}.npy'
        shear_array = self._array_from_message(shear_msg)
        np.save(str(shear_path), shear_array)

        entry = [
            idx,
            timestamp,
            json.dumps(wrench_values),
            str(shear_path.relative_to(self.output_dir)),
        ]
        with self.index_file.open('a', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(entry)
        return idx

    def _reserve_index(self) -> int:
        with self._lock:
            idx = self._next_index
            self._next_index += 1
        return idx

    @staticmethod
    def _choose_timestamp(wrench_msg: WrenchStamped, shear_msg: Float32MultiArrayStamped) -> int:
        for header in (wrench_msg.header, shear_msg.header):
            stamp = getattr(header, 'stamp', rospy.Time())
            if stamp and stamp.to_sec() > 0.0:
                return stamp.to_nsec()
        return rospy.Time.now().to_nsec()

    @staticmethod
    def _wrench_to_list(msg: WrenchStamped) -> List[float]:
        wrench = msg.wrench
        return [
            float(wrench.force.x),
            float(wrench.force.y),
            float(wrench.force.z),
            float(wrench.torque.x),
            float(wrench.torque.y),
            float(wrench.torque.z),
        ]

    @staticmethod
    def _array_from_message(msg: Float32MultiArrayStamped) -> np.ndarray:
        array = np.asarray(msg.data, dtype=np.float32)
        dims = [dim.size for dim in msg.layout.dim if dim.size]
        total = 1
        for size in dims:
            total *= size
        if dims and total == array.size:
            array = array.reshape(dims)
        return array


def main() -> None:
    rospy.init_node('single_shot_dataset_recorder', anonymous=False)
    SingleShotRecorder()
    rospy.spin()


if __name__ == '__main__':
    main()

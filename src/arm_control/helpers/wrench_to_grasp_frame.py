#!/usr/bin/env python3
import rospy
import numpy as np
import tf2_ros
from geometry_msgs.msg import WrenchStamped
from tf.transformations import quaternion_matrix


class WrenchProjector:
    def __init__(self):
        self.target_frame = rospy.get_param('~target_frame', 'grasp_frame')
        self.input_topic = rospy.get_param('~input_topic', '/netft/netft_data')
        self.output_topic = rospy.get_param('~output_topic', '/wrench_grasp_frame')

        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.pub = rospy.Publisher(self.output_topic, WrenchStamped, queue_size=10)
        self.sub = rospy.Subscriber(self.input_topic, WrenchStamped, self.cb, queue_size=10)

        rospy.loginfo("WrenchProjector: projecting %s to %s -> %s",
                      self.input_topic, self.target_frame, self.output_topic)

    @staticmethod
    def _skew(v):
        return np.array([[0, -v[2], v[1]],
                         [v[2], 0, -v[0]],
                         [-v[1], v[0], 0]])

    def cb(self, msg: WrenchStamped):
        src_frame = msg.header.frame_id
        if not src_frame:
            return
        try:
            # Transform from source -> target: x_T = R x_S + t
            tf: tf2_ros.TransformStamped = self.tf_buffer.lookup_transform(
                self.target_frame, src_frame, rospy.Time(0), timeout=rospy.Duration(0.1))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logwarn_throttle(5.0, 'TF lookup failed %s->%s: %s', src_frame, self.target_frame, str(e))
            return

        t = np.array([tf.transform.translation.x,
                      tf.transform.translation.y,
                      tf.transform.translation.z])
        q = [tf.transform.rotation.x,
             tf.transform.rotation.y,
             tf.transform.rotation.z,
             tf.transform.rotation.w]
        R = quaternion_matrix(q)[:3, :3]  # R_T_S
        Rst = R.T  # R_S_T

        F_S = np.array([msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z])
        tau_S = np.array([msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z])

        # r_S is vector from S-origin to T-origin, expressed in S
        r_S = - Rst.dot(t)

        # Transform wrench
        F_T = -R.dot(F_S)
        tau_T = -R.dot(tau_S + np.cross(r_S, F_S))

        out = WrenchStamped()
        out.header.stamp = msg.header.stamp if msg.header.stamp != rospy.Time() else rospy.Time.now()
        out.header.frame_id = self.target_frame
        out.wrench.force.x, out.wrench.force.y, out.wrench.force.z = F_T.tolist()
        out.wrench.torque.x, out.wrench.torque.y, out.wrench.torque.z = tau_T.tolist()
        self.pub.publish(out)


def main():
    rospy.init_node('wrench_to_grasp_frame')
    _ = WrenchProjector()
    rospy.spin()


if __name__ == '__main__':
    main()


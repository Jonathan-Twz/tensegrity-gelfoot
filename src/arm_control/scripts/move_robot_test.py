import rospy
from bubble_utils.bubble_med.bubble_med import BubbleMed

from victor_hardware_interface_msgs.msg import ControlMode

from victor_hardware_interface import victor_utils as vu
import geometry_msgs.msg
from std_msgs.msg import Bool

from mmint_utils.gamma_helpers.gamma_helpers import zero_ati_gamma

distal_frame = 'grasp_frame'

robot = BubbleMed()

# hardcoded home joint positions for BubbleMed
home_joint_positions = [-1.5571975271691387, -0.21685263217621578, -0.5989819870530589, 1.6159687053844756, 1.5529173013465019, -1.065497063113712, -1.0476360740068908]

# netural pose
neutural_pose = [ 0.1268571, 0.58309891, 0.6358351, -0.59653931, -0.50834194, -0.13021979, 0.60726612]

# wrench_topic (geometry_msgs/WrenchStamped) to read forces from
wrench_topic = '/wrench_grasp_frame'

# flag_topic (std_msgs/Bool) to start and stop recording
flag_topic = '/dataset/is_recording'
pub_flag = rospy.Publisher(flag_topic, Bool, queue_size=1)


def go_back_to_home():
    print("Going back to home joint positions...")
    # change robot control mode to joint position
    robot_control_mode = ControlMode.JOINT_POSITION
    robot.set_control_mode(robot_control_mode, vel=0.02)
    robot.set_joint_position_control(vel=0.02)
    robot.plan_to_joint_config(robot.arm_group, home_joint_positions)
    # robot.set_joint_positions(home_joint_positions, vel=0.02)
    current_joint_positions = robot.get_joint_positions(robot.get_joint_names(robot.arm_group))
    print("Current joint positions:", current_joint_positions)

def get_current_wrench_once():
    # get current wrench once from the wrench topic, read the force and torque values from the topic, convert to numpy array and return
    wrench_msg = rospy.wait_for_message(wrench_topic, geometry_msgs.msg.WrenchStamped)
    force = wrench_msg.wrench.force
    torque = wrench_msg.wrench.torque
    print("Current force: ", force)
    print("Current torque: ", torque)
    wrench = [force.x, force.y, force.z, torque.x, torque.y, torque.z]
    return wrench

def _print_plan_info(result):
    try:
        plan = result.planning_result.plan
        pts = plan.joint_trajectory.points
        npts = len(pts)
        print(f"Plan points: {npts}")
        if npts >= 1:
            start = pts[0].positions
            end = pts[-1].positions
            import numpy as np
            d = np.max(np.abs(np.array(end) - np.array(start)))
            print(f"Max joint delta in plan: {d:.6f} rad")
    except Exception as e:
        print(f"No plan info available: {e}")


def make_contact_down_force(max_force = 30.0, min_force = 0.5):
    
    # first go back home
    go_back_to_home()

    # import pdb; pdb.set_trace()
    rospy.sleep(1.0)
    # then use joint impedance control to make contact with the table
    # robot_control_mode = ControlMode.JO
    # then move down 1mm per step, and also read the force from the wrench topic, get the normalized force, compare the force with max_force, if the force is greater than max_force, stop moving
    # robot_control_mode = ControlMode.JOINT_IMPEDANCE
    # stiffness = vu.Stiffness.MEDIUM #also available: (SOFT?), MEDIUM, STIFF
    # robot.set_control_mode(robot_control_mode, stiffness=stiffness, vel=0.01, accel=0.5)
    current_pose = robot.get_current_pose()
    print("Current pose: ", current_pose)
    pose = current_pose

    # zero ATI gamma
    zero_ati_gamma()

    rate = rospy.Rate(2) # 2 Hz for quicker feedback
    while not rospy.is_shutdown():
        wrench = get_current_wrench_once()
        force = geometry_msgs.msg.Vector3()
        force.x = wrench[0]
        force.y = wrench[1]
        force.z = wrench[2]
        norm_force = (force.x**2 + force.y**2 + force.z**2)**0.5
        print("Current force: ", norm_force)
        if norm_force >= max_force:
            print("Max force reached, stopping...")
            break
        elif norm_force < min_force:
            print("Force too low, moving down...")
        else:
            print("Force within range, moving down slowly...")
            # start the flag to record dataset
            pub_flag.publish(True)
        pose[2] -= 0.0002 # move down 1 mm
        goal_pose = pose
        print("Moving to pose: ", goal_pose)
        # Log IK result and planned trajectory details for debugging tiny-step behavior
        try:
            import numpy as np
            current_joints = robot.get_joint_positions(robot.get_joint_names(robot.arm_group))
        except Exception:
            current_joints = None
        try:
            ik_joints, ik_err = robot.compute_ik(list(goal_pose), ee_link_name=distal_frame, ref_frame='med_base')
            if hasattr(ik_err, 'val'):
                print(f"IK err code: {ik_err.val}")
            if current_joints is not None:
                jd = np.max(np.abs(np.array(ik_joints) - np.array(current_joints)))
                print(f"Max IK joint delta: {jd:.6f} rad")
        except Exception as e:
            print(f"IK call failed: {e}")

        res = robot.set_pose(pose=goal_pose, frame_id=distal_frame, ref_frame='med_base', position_tol=0.0005)
        _print_plan_info(res)
        rate.sleep()
        # import pdb; pdb.set_trace()
    current_pose = robot.get_current_pose()
    print("Final pose after making contact: ", current_pose)
    go_back_to_home()
    rospy.sleep(0.2)


# get current joint positions
# # BubbleMed exposes get_joint_positions(), not get_current_joint_positions()
# joint_positions = robot.get_joint_positions(robot.get_joint_names(robot.arm_group))
# print("Current joint positions:", joint_positions)

robot_control_mode = ControlMode.JOINT_POSITION

robot_vel = 5 # mm/s

robot.set_control_mode(robot_control_mode, vel=robot_vel*0.01)

#robot_control_mode = ControlMode.JOINT_IMPEDANCE
#stiffness = vu.Stiffness.MEDIUM #also available: (SOFT?), MEDIUM, STIFF
#robot.set_control_mode(robot_control_mode, stiffness=stiffness, vel=0.25, accel=0.5)
# get current pose
current_pose = robot.get_current_pose()
print("Current pose: ", current_pose)
pose = current_pose

make_contact_down_force(max_force = 15.0)
#

# pose is a list of 7 values: [x, y, z, qx, qy, qz, qw]
# pose[2] += 0.001 # move up 10 cm
# import pdb; pdb.set_trace()
# robot.set_pose(pose=pose, frame_id=distal_frame)
# robot.set_raw_pose(pose=pose, frame_id=distal_frame)
# current_pose = robot.get_current_pose()

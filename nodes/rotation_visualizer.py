import numpy as np

import rospy
import tf2_ros

from tf.transformations import quaternion_from_euler
from geometry_msgs.msg import TransformStamped, Quaternion, Point



class Rotator:
    def __init__(self):
        pass

        self.tf_pub = tf2_ros.TransformBroadcaster()

    def publish_transform(self, pos, rot):

        x,y,z = pos
        ax, ay, az = rot

        # Normalize the position, so that the new system is 1m apart
        norm = np.linalg.norm([x, y, z])
        x, y, z = x/norm, y/norm, z/norm

        # Get quaternion from the rotation angles
        quat_list = quaternion_from_euler(ax, ay, az)
        quat = Quaternion(*quat_list)
        
        tf_msg = TransformStamped()
        tf_msg.header.stamp = rospy.Time.now()
        tf_msg.header.frame_id = 'ECEF'
        tf_msg.child_frame_id = 'camera'
        tf_msg.transform.translation.x = x
        tf_msg.transform.translation.y = y
        tf_msg.transform.translation.z = z
        tf_msg.transform.rotation = quat

        self.tf_pub.sendTransform(tf_msg)


if __name__ == "__main__":

    rospy.init_node("ecef_sim")

    rt = Rotator()
    r = 1

    pos = [4366651.806791375, 954196.6482927991, 4540212.303351457]
    rot = [168.1311312733719 -43.265336941771466 -107.04773794209594]
    rot = np.radians(rot)

    while not rospy.is_shutdown():
        rt.publish_transform()
        r.sleep()
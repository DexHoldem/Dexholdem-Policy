import rospy
from sr_robot_commander.sr_hand_commander import SrHandCommander
from sr_utilities.hand_finder import HandFinder
import time
import numpy as np
import pyrealsense2 as rs
import cv2
import argparse
import os
import glob
from copy import deepcopy

# Argument parser
parser = argparse.ArgumentParser(description="Shadow Hand + RealSense data collector")
parser.add_argument("--save_dir", type=str, required=True, help="Directory to save data")
args = parser.parse_args()

# Ensure directories
os.makedirs(args.save_dir, exist_ok=True)
image_dirs = []
depth_dirs = []

for i in range(3):
    img_dir = os.path.join(args.save_dir, f"images_cam{i}")
    dpt_dir = os.path.join(args.save_dir, f"depth_cam{i}")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(dpt_dir, exist_ok=True)
    image_dirs.append(img_dir)
    depth_dirs.append(dpt_dir)

# Get next available index
def get_next_index(folder):
    files = glob.glob(os.path.join(folder, "data_*.npz"))
    if not files:
        return 1
    indices = [int(os.path.splitext(os.path.basename(f))[0].split("_")[-1]) for f in files]
    return max(indices) + 1

# ROS and hand commander
rospy.init_node("shadow_hand_data_collector", anonymous=True)


hand_finder = HandFinder()
hand_parameters = hand_finder.get_hand_parameters()

hand_commander = SrHandCommander("right_hand")
hand_commander.refresh_named_targets()

# RealSense setup
ctx = rs.context()
devices = ctx.query_devices()
if len(devices) < 3:
    raise RuntimeError(f"Expected at least 3 RealSense cameras, found {len(devices)}.")

serials = [dev.get_info(rs.camera_info.serial_number) for dev in devices[:3]]
pipelines = []
align_processors = []

for serial in serials:
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
    pipeline.start(config)
    pipelines.append(pipeline)
    align_processors.append(rs.align(rs.stream.color))  # Align depth to color

print(f"Started RealSense camera: {serial}")

# Buffers
joint_positions = []
joint_velocities = []
joint_efforts = []
tactile_states = []

images = [[] for _ in range(3)]
depth_images = [[] for _ in range(3)]

for _ in range(15):
    for i, pipeline in enumerate(pipelines):
        pipeline.wait_for_frames()

index = get_next_index(args.save_dir)
print("Collecting data at ~15Hz from 3 RealSense cameras. Press Ctrl+C to stop and save...")

def collect_data():
    global index
    global images
    global depth_images
    try:
        while not rospy.is_shutdown():
            for i, pipeline in enumerate(pipelines):
                frames = pipeline.wait_for_frames()
                aligned_frames = align_processors[i].process(frames)
                depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()

                if not depth_frame or not color_frame:
                    continue

                # Get color and depth images
                color_img = np.asanyarray(color_frame.get_data()).copy()
                depth_img = np.asanyarray(depth_frame.get_data()).copy()

                # Keep RealSense bgr8 channel order.  The TexasPokerRobot
                # training data stores images_cam* as BGR arrays.

                # ✅ Append image and depth to buffers
                images[i].append(color_img)
                print(len(images[i]))
                depth_images[i].append(depth_img)
                print(len(depth_images[i]))

            # Get Shadow Hand data
            joints_position = hand_commander.get_joints_position()
            joints_velocity = hand_commander.get_joints_velocity()
            joints_effort = hand_commander.get_joints_effort()
            # tactile_state = hand_commander.get_tactile_state()
            # print(tactile_state)
            joint_positions.append(joints_position)
            joint_velocities.append(joints_velocity)
            joint_efforts.append(joints_effort)
            # tactile_states.append(tactile_state)

            time.sleep(1 / 15.0)

    except KeyboardInterrupt:
        print("\nData collection stopped. Saving data...")
    finally:
        for p in pipelines:
            p.stop()
        rospy.signal_shutdown("User stop")

        user_input = input("Do you want to save the data? (y/n): ").strip().lower()
        if user_input != 'y':
            print("Data was not saved.")
            return

        # Save combined npz
        output_path = os.path.join(args.save_dir, f"data_{index:04d}.npz")
        np.savez_compressed(output_path,
                            joint_positions=np.array(joint_positions),
                            joint_velocities=np.array(joint_velocities),
                            joint_efforts=np.array(joint_efforts),
                            # tactile_states=np.array(tactile_states),
                            images_cam0=np.array(images[0]),
                            images_cam1=np.array(images[1]),
                            images_cam2=np.array(images[2]),
                            depth_cam0=np.array(depth_images[0]),
                            depth_cam1=np.array(depth_images[1]),
                            depth_cam2=np.array(depth_images[2]))
        print(f"Saved combined data to: {output_path}")

        # Save individual PNGs for images and depth
        for i in range(3):
            for idx, (color_img, depth_img) in enumerate(zip(images[i], depth_images[i])):
                color_path = os.path.join(image_dirs[i], f"cam{i}_image_{idx:04d}.png")
                depth_path = os.path.join(depth_dirs[i], f"cam{i}_depth_{idx:04d}.png")

                # color_img is already BGR, which OpenCV expects.
                cv2.imwrite(color_path, color_img)

                # Normalize depth for visualization
                depth_img_normalized = cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX)
                depth_img_normalized = depth_img_normalized.astype(np.uint8)

                # Save raw 16-bit depth image
                raw_depth_path = os.path.join(depth_dirs[i], f"cam{i}_depth_raw_{idx:04d}.png")
                cv2.imwrite(raw_depth_path, depth_img)

                # Save normalized depth image
                cv2.imwrite(depth_path, depth_img_normalized)

        print("Saved all color and depth images to disk.")

if __name__ == "__main__":
    collect_data()

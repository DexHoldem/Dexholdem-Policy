import argparse
import json
import time
import zmq
import numpy as np
import cv2
import pyrealsense2 as rs
import rospy
import threading
import queue
import collections
from collections import deque
from sr_robot_commander.sr_hand_commander import SrHandCommander
from sr_utilities.hand_finder import HandFinder



class RobotClient:
    JOINT_ORDER = [
        # 手臂关节 (0-5)
        'ra_shoulder_pan_joint', 'ra_shoulder_lift_joint', 'ra_elbow_joint',
        'ra_wrist_1_joint', 'ra_wrist_2_joint', 'ra_wrist_3_joint',
        # 手部关节 (6-29)
        'rh_FFJ1', 'rh_FFJ2', 'rh_FFJ3', 'rh_FFJ4',
        'rh_MFJ1', 'rh_MFJ2', 'rh_MFJ3', 'rh_MFJ4',
        'rh_RFJ1', 'rh_RFJ2', 'rh_RFJ3', 'rh_RFJ4',
        'rh_LFJ1', 'rh_LFJ2', 'rh_LFJ3', 'rh_LFJ4', 'rh_LFJ5',
        'rh_THJ1', 'rh_THJ2', 'rh_THJ3', 'rh_THJ4', 'rh_THJ5',
        'rh_WRJ1', 'rh_WRJ2'
    ]

    def __init__(self, server_ip="localhost", port=13579, instruction=None, obs_horizon=1):
        # 初始化ROS
        rospy.init_node("robot_client", anonymous=True)
        
        # 初始化机器人控制
        self.hand_finder = HandFinder()
        self.hand_parameters = self.hand_finder.get_hand_parameters()
        self.hand_commander = SrHandCommander("right_hand")
        self.hand_commander.refresh_named_targets()
        
        # 初始化ZeroMQ
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        
        # 使用WiFi接口连接服务器
        server_address = f"tcp://{server_ip}:{port}"
        print(f"Connecting to server at {server_address}")
        self.socket.connect(server_address)
        # Set recv timeout so client doesn't hang if server crashes (10 seconds).
        self.socket.setsockopt(zmq.RCVTIMEO, 10000)
        print("Connected to server")
        
        # 获取服务器配置（简化版本）
        self.obs_horizon = obs_horizon
        self.action_horizon = 32  # 默认值
        self.instruction_dim = 1  # 默认值
        # Model always predicts absolute joint positions (not deltas).
        self.clip_depth_max = False  # 默认值：是否裁剪深度
        self.depth_max_value = 4000.0  # 默认深度裁剪上限（mm）
        self.camera_indices = [0, 1, 2]  # 默认值：所有相机（如果没有从服务器获取配置）
        
        # 🔧 先获取服务器配置（包括instruction_dim和camera_indices），再处理instruction和初始化相机
        # 初始化相机列表（将在_get_server_config后初始化）
        self.pipelines = []
        self.align_processors = []
        
        # 🔧 先获取服务器配置（需要instruction_dim来正确处理instruction）
        self._get_server_config()
        
        # 🔧 获取配置后再处理instruction（需要正确的instruction_dim）
        self.instruction = self._process_instruction_input(instruction)
        print(f"设置任务instruction: {self.instruction}")
        
        # 初始化相机（使用从服务器获取的camera_indices）
        self._init_cameras()
        
        # observation队列
        self.obs_deque = None
    
    def _process_instruction_input(self, instruction):
        """处理instruction输入，转换为模型期望的格式"""
        if instruction is None:
            instruction = 0
        
        # 使用与训练时相同的instruction_to_vector逻辑
        instruction_vector = self._instruction_to_vector(instruction, self.instruction_dim)
        print(f"📝 处理instruction: {instruction} -> {instruction_vector}")
        return instruction_vector
    
    def _instruction_to_vector(self, instruction_num, instruction_dim=1):
        """将instruction数字转换为one-hot向量表示"""
        # 创建one-hot向量
        vector = np.zeros(instruction_dim, dtype=np.float32)
        
        # 确保instruction_num在有效范围内
        if instruction_num < 0:
            print(f"警告: instruction_num ({instruction_num}) < 0，使用0")
            instruction_num = 0
        elif instruction_num >= instruction_dim:
            print(f"警告: instruction_num ({instruction_num}) >= instruction_dim ({instruction_dim})，使用取模")
            instruction_num = instruction_num % instruction_dim
        
        # 设置one-hot编码
        vector[instruction_num] = 1.0
        
        print(f"🔢 机器人客户端: Instruction {instruction_num} -> one-hot向量: {vector}")
        return vector
    
    def _get_server_config(self):
        """从服务器获取配置参数（简化版本）"""
        try:
            print("🔧 获取服务器配置...")
            # 发送配置请求
            config_request = {
                'type': 'config_request',
                'timestamp': time.time()
            }
            self.socket.send_json(config_request)
            
            # 接收配置响应（use longer timeout for initial handshake）
            self.socket.setsockopt(zmq.RCVTIMEO, 30000)
            config_response = self.socket.recv_json()
            self.socket.setsockopt(zmq.RCVTIMEO, 10000)
            
            if 'config' in config_response:
                config = config_response['config']
                self.obs_horizon = config.get('obs_horizon', self.obs_horizon)
                self.action_horizon = config.get('action_horizon', self.action_horizon)
                use_instruction = config.get('use_instruction', False)
                self.instruction_dim = config.get('instruction_dim', 1) if use_instruction else 1
                # predict_pos_delta is ignored — model always predicts absolute positions
                self.clip_depth_max = config.get('clip_depth_max', self.clip_depth_max)
                self.depth_max_value = config.get('depth_max_value', self.depth_max_value)
                # 🔧 获取camera_indices配置
                if 'camera_indices' in config:
                    self.camera_indices = config['camera_indices']
                
                print(f"📋 服务器配置:")
                print(f"  obs_horizon: {self.obs_horizon}")
                print(f"  action_horizon: {self.action_horizon}")
                print(f"  use_instruction: {use_instruction}")
                print(f"  instruction_dim: {self.instruction_dim}")
                print(f"  action_mode: absolute positions")
                print(f"  clip_depth_max: {self.clip_depth_max}, depth_max_value: {self.depth_max_value}")
                print(f"  camera_indices: {self.camera_indices}")
                
            else:
                print("⚠️  服务器未返回配置，使用默认值")
                
        except Exception as e:
            print(f"❌ 获取服务器配置失败: {e}")
            print("⚠️  使用默认配置")
    
    def _init_cameras(self):
        # 初始化RealSense相机
        ctx = rs.context()
        devices = ctx.query_devices()
        
        # 🔧 使用配置的camera_indices，支持不同数量的相机
        max_cam_idx = max(self.camera_indices) if self.camera_indices else 0
        required_cameras = max_cam_idx + 1
        
        if len(devices) < required_cameras:
            raise RuntimeError(
                f"Expected at least {required_cameras} RealSense cameras (based on camera_indices {self.camera_indices}), "
                f"found {len(devices)}."
            )

        # 🔧 只获取指定索引的相机
        serials = []
        for cam_idx in self.camera_indices:
            if cam_idx >= len(devices):
                raise RuntimeError(f"Camera index {cam_idx} not available. Only {len(devices)} cameras found.")
            serial = devices[cam_idx].get_info(rs.camera_info.serial_number)
            serials.append(serial)
        
        # 为每个指定相机创建pipeline和配置
        for serial in serials:
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_device(serial)
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
            pipeline.start(config)
            self.pipelines.append(pipeline)
            self.align_processors.append(rs.align(rs.stream.color))  # 将深度图对齐到彩色图
            
        print(f"Started RealSense cameras (indices {self.camera_indices}): {serials}")
        
        # 等待相机初始化
        for _ in range(15):
            for pipeline in self.pipelines:
                pipeline.wait_for_frames()
    
    def _get_observation(self):
        # 获取当前状态的observation
        obs = {}
        
        # 获取图像数据
        rgb_images = []
        depth_images = []
                
        for i, pipeline in enumerate(self.pipelines):
            frames = pipeline.wait_for_frames()
            aligned_frames = self.align_processors[i].process(frames)
            
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            
            if not depth_frame or not color_frame:
                continue
                
            # 获取彩色和深度图像
            color_img = np.asanyarray(color_frame.get_data()).copy()
            depth_img = np.asanyarray(depth_frame.get_data()).copy()

            # Keep RealSense bgr8 channel order to match TexasPokerRobot
            # training data, where images_cam* are stored as BGR arrays.

            if self.clip_depth_max:
                depth_img = np.clip(depth_img, 0, self.depth_max_value)
            
            rgb_images.append(color_img)
            depth_images.append(depth_img)
        
        if rgb_images and depth_images:
            # 🔧 使用配置的camera_indices存储数据，支持不同数量的相机
            for i, cam_idx in enumerate(self.camera_indices):
                obs[f'images_cam{cam_idx}'] = rgb_images[i]  # (H, W, 3)
                obs[f'depth_cam{cam_idx}'] = depth_images[i]  # (H, W)
        
        # 获取机器人状态数据
        joints_position = self.hand_commander.get_joints_position()
        joints_effort = self.hand_commander.get_joints_effort()
        joints_velocity = self.hand_commander.get_joints_velocity()
        
        # 按顺序获取关节位置数据
        joint_positions_list = []
        for joint_name in self.JOINT_ORDER:
            if joint_name in joints_position:
                joint_positions_list.append(joints_position[joint_name])
            else:
                joint_positions_list.append(0.0)  # 缺失关节用0填充
        
        # 按顺序获取关节努力值数据
        joint_efforts_list = []
        for joint_name in self.JOINT_ORDER:
            if joint_name in joints_effort:
                joint_efforts_list.append(joints_effort[joint_name])
            else:
                joint_efforts_list.append(0.0)  # 缺失关节用0填充
        # 按顺序获取关节速度数据
        joint_velocities_list = []
        for joint_name in self.JOINT_ORDER:
            if joint_name in joints_velocity:
                joint_velocities_list.append(joints_velocity[joint_name])
            else:
                joint_velocities_list.append(0.0)  # 缺失关节用0填充
        
        # 转换为numpy数组
        joint_positions_np = np.array(joint_positions_list)
        joint_efforts_np = np.array(joint_efforts_list)
        joint_velocities_np = np.array(joint_velocities_list)
        
        # 存储关节数据
        obs['joint_positions'] = joint_positions_np
        obs['joint_efforts'] = joint_efforts_np
        obs['joint_velocities'] = joint_velocities_np
        
        # 添加instruction数据（任务级别，已经是one-hot向量形式）
        obs['instruction'] = self.instruction
        
        return obs
    
    def _execute_action(self, action_to_execute):
        # action_to_execute is expected to be a 1D numpy array of 30 absolute joint positions

        print(f"Executing action (absolute positions): {action_to_execute}")

        if len(action_to_execute) != len(self.JOINT_ORDER):
            print(f"Error: Action length ({len(action_to_execute)}) does not match JOINT_ORDER length ({len(self.JOINT_ORDER)}).")
            return

        # 创建目标位置字典 — 模型直接预测绝对位置
        target_positions = {}

        for i, joint_name in enumerate(self.JOINT_ORDER):
            target_positions[joint_name] = float(action_to_execute[i])
            print(f"Joint {joint_name}: -> {action_to_execute[i]:.4f}")
        
        print(f"Target positions: {target_positions}")
        
        # 执行动作
        try:
            # Parameters match the offline playback client.
            self.hand_commander.move_to_joint_value_target_unsafe(
                target_positions, 
                wait=True, 
                time=1.0,
                angle_degrees=False
            )
            print("Movement command sent successfully and completed.")
        except Exception as e:
            print(f"Error executing movement: {str(e)}")
            raise
    
    def run(self):
        print("Robot client started")
        print("=" * 50)
        print(f"当前任务instruction: {self.instruction}")
        print(f"时序配置: obs_horizon={self.obs_horizon}, action_horizon={self.action_horizon}")
        print(f"动作模式: 绝对位置")
        print("=" * 50)
        
        while not rospy.is_shutdown():
            try:
                # 获取observation
                print("\n🔍 Getting observation...")
                obs = self._get_observation()
                print(f"✅ Got observation, 包含instruction: {obs['instruction']}")

                # 初始化或更新observation队列
                if self.obs_deque is None:
                    print("Initializing observation queue...")
                    self.obs_deque = collections.deque(maxlen=self.obs_horizon)
                    # 用当前的observation填充队列
                    for _ in range(self.obs_horizon):
                        self.obs_deque.append(obs)
                    print("Observation queue initialized with current observation")
                else:
                    # 更新队列：deque的maxlen自动丢弃最旧的observation
                    self.obs_deque.append(obs)
                    print("Observation queue updated with new observation")
                
                # 将NumPy数组转换为Python列表
                obs_list = []
                for obs_in_queue in self.obs_deque:
                    json_obs = {}
                    for key, value in obs_in_queue.items():
                        if isinstance(value, np.ndarray):
                            json_obs[key] = value.tolist()
                        else:
                            json_obs[key] = value
                    obs_list.append(json_obs)
                
                # 发送observation到服务器
                message = {
                    'observation': obs_list,
                    'timestamp': time.time()
                }
                print("📤 Sending observation to server...")
                self.socket.send_json(message)
                
                # 接收action
                print("📥 Waiting for action from server...")
                try:
                    response = self.socket.recv_json()
                except zmq.Again:
                    print("❌ Server response timed out — retrying next cycle.")
                    continue
                print(f"Received response: {response.keys()}")
                
                if 'error' in response:
                    print(f"❌ Error from server: {response['error']}")
                    continue
                
                # 处理action序列
                action_sequence = np.array(response['action'])
                print(f"📊 Received action sequence shape: {action_sequence.shape}")
                
                # 执行action序列
                if action_sequence.ndim == 2 and action_sequence.shape[0] > 0:
                    num_actions = action_sequence.shape[0]
                    print(f"🚀 执行 {num_actions} 个action步骤")
                    
                    for i in range(num_actions):
                        if rospy.is_shutdown():
                            print("🛑 ROS shutdown requested during action execution.")
                            break
                        
                        action_to_execute = action_sequence[i]
                        print(f"🎯 执行action {i+1}/{num_actions}")


                        # 获取observation
                        print("\n🔍 Getting observation...")
                        obs = self._get_observation()
                        print(f"✅ Got observation, 包含instruction: {obs['instruction']}")

                        self.obs_deque.append(obs)
                        print("Observation queue updated with new observation")
                        
                        # 执行action
                        self._execute_action(action_to_execute)
                    
                    print(f"✅ Action序列执行完成")
                    
                elif action_sequence.ndim == 1:
                    # 单步action：直接执行
                    print("🎯 执行单步action")
                    self._execute_action(action_sequence)
                    
                else:
                    print(f"❌ Unexpected action shape: {action_sequence.shape}")
                    continue
                
            except Exception as e:
                print(f"❌ Error in main loop: {str(e)}")
                print(f"Error type: {type(e)}")
                import traceback
                print(f"Traceback: {traceback.format_exc()}")
                time.sleep(1)  # 发生错误时等待一段时间再重试
    
    def cleanup(self):
        # 清理资源
        for pipeline in self.pipelines:
            pipeline.stop()
        self.socket.close()
        self.context.term()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--server_ip', type=str, default="192.168.1.200",
                      help='IP address of the policy server')
    parser.add_argument('--port', type=int, default=13579,
                      help='ZeroMQ port number')
    parser.add_argument('--instruction', type=int, default=None,
                      help='Instruction for the task as single integer (e.g., 0, 1, 2, 3)')
    parser.add_argument('--obs_horizon', type=int, default=1,
                      help='Observation horizon')
    args = parser.parse_args()
    
    # 直接使用instruction参数（argparse已经解析为整数）
    instruction = args.instruction
    
    print(f"🎮 启动机器人客户端")
    print(f"📝 Instruction: {instruction}")
    print(f"📊 观测窗口: {args.obs_horizon}")
    
    client = RobotClient(args.server_ip, args.port, instruction, args.obs_horizon)
    try:
        client.run()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        client.cleanup()

if __name__ == "__main__":
    main() 

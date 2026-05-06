import os
import numpy as np
from pathlib import Path

def read_npz_file(file_path):
    """
    读取npz文件并返回其内容
    
    Args:
        file_path: npz文件路径
    Returns:
        dict: 包含npz文件中所有数组的字典
    """
    try:
        data = np.load(file_path, allow_pickle=True)
        return {key: data[key] for key in data.files}
    except Exception as e:
        print(f"读取文件 {file_path} 时出错: {str(e)}")
        return None

def print_data_info(data_dict):
    """
    打印数据字典中每个数组的信息
    
    Args:
        data_dict: 包含数据的字典
    """
    if data_dict is None:
        return
    
    print("\n数据内容:")
    print("-" * 50)
    for key, value in data_dict.items():
        print(f"\n键名: {key}")
        print(f"类型: {type(value)}")
        print(f"形状: {value.shape}")
        if isinstance(value, np.ndarray):
            if value.dtype.kind in 'iuf':  # 整数或浮点数
                print(f"数据类型: {value.dtype}")
                print(f"最小值: {np.min(value)}")
                print(f"最大值: {np.max(value)}")
                print(f"平均值: {np.mean(value)}")
                
                # 特殊处理关节位置数据
                if key == 'joint_positions':
                    if len(value.shape) == 1:  # 单个时间步
                        print("\n关节位置数据:")
                        print(f"  手臂关节(0-5): {value[:6]}")
                        print(f"  手部关节(6-29): {value[6:]}")
                    elif len(value.shape) == 2:  # 多个时间步
                        print("\n关节位置数据(第一个时间步):")
                        print(f"  手臂关节(0-5): {value[0, :6]}")
                        print(f"  手部关节(6-29): {value[0, 6:]}")
            elif value.dtype.kind == 'O':  # 对象数组
                print("数据类型: object")
                print("内容示例:")
                if len(value) > 0:
                    if isinstance(value[0], dict):
                        # 如果是字典格式的关节位置数据
                        print("关节位置数据:")
                        arm_joints = ['ra_shoulder_pan_joint', 'ra_shoulder_lift_joint', 'ra_elbow_joint',
                                    'ra_wrist_1_joint', 'ra_wrist_2_joint', 'ra_wrist_3_joint']
                        hand_joints = ['rh_FFJ1', 'rh_FFJ2', 'rh_FFJ3', 'rh_FFJ4',
                                     'rh_MFJ1', 'rh_MFJ2', 'rh_MFJ3', 'rh_MFJ4',
                                     'rh_RFJ1', 'rh_RFJ2', 'rh_RFJ3', 'rh_RFJ4',
                                     'rh_LFJ1', 'rh_LFJ2', 'rh_LFJ3', 'rh_LFJ4', 'rh_LFJ5',
                                     'rh_THJ1', 'rh_THJ2', 'rh_THJ3', 'rh_THJ4', 'rh_THJ5',
                                     'rh_WRJ1', 'rh_WRJ2']
                        
                        print("  手臂关节:")
                        for joint in arm_joints:
                            if joint in value[0]:
                                print(f"    {joint}: {value[0][joint]}")
                        
                        print("  手部关节:")
                        for joint in hand_joints:
                            if joint in value[0]:
                                print(f"    {joint}: {value[0][joint]}")
                    else:
                        print(value[0])
        print("-" * 30)

def main():
    # 设置数据集路径
    dataset_path = "data/pick_up_left_0530"
    
    # 检查路径是否存在
    if not os.path.exists(dataset_path):
        print(f"错误: 路径 '{dataset_path}' 不存在")
        return
    
    # 获取所有npz文件
    npz_files = list(Path(dataset_path).rglob("*.npz"))
    
    if not npz_files:
        print(f"在 '{dataset_path}' 中没有找到npz文件")
        return
    
    print(f"找到 {len(npz_files)} 个npz文件")
    
    # 读取并显示每个文件的内容
    for i, file_path in enumerate(npz_files, 1):
        print(f"\n处理文件 {i}/{len(npz_files)}: {file_path}")
        data = read_npz_file(str(file_path))
        if data:
            print_data_info(data)
        
        # 每处理1个文件后询问是否继续
        if i % 1 == 0 and i < len(npz_files):
            response = input("\n是否继续显示下一个文件? (y/n): ")
            if response.lower() != 'y':
                break

if __name__ == "__main__":
    main() 
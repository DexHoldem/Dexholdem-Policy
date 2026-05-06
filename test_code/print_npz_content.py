#!/usr/bin/env python3
"""
打印指定npz文件中的joints position信息
"""
import numpy as np
import os
import sys
from pathlib import Path

def print_joints_position(npz_path):
    """打印npz文件中的joints position信息"""
    if not os.path.exists(npz_path):
        print(f"错误：文件不存在: {npz_path}")
        return False
    
    try:
        # 读取npz文件
        data = np.load(npz_path, allow_pickle=True)
        
        print(f"=== 文件: {npz_path} ===")
        
        # 查找joints position相关的键
        joints_keys = []
        for key in data.files:
            # 查找可能的joints position键名
            if any(keyword in key.lower() for keyword in ['joint', 'qpos', 'position', 'pos']):
                joints_keys.append(key)
        
        if not joints_keys:
            print("未找到joints position相关的数据")
            print(f"文件中包含的键: {sorted(data.files)}")
            return False
        
        # 打印找到的joints position数据
        for key in sorted(joints_keys):
            arr = data[key]
            print(f"\n键名: '{key}'")
            print(f"形状: {arr.shape}")
            print(f"数据类型: {arr.dtype}")
            
            if np.issubdtype(arr.dtype, np.number):
                print(f"数值范围: [{arr.min():.4f}, {arr.max():.4f}]")
            
            # 显示具体的joints position数据
            if len(arr.shape) == 1:
                # 单个时间步的joints position
                print(f"joints position: {arr}")
            elif len(arr.shape) == 2:
                # 时间序列的joints position
                print(f"时间步数: {arr.shape[0]}, 关节数: {arr.shape[1]}")
                print("前5个时间步的joints position:")
                for i in range(min(5, arr.shape[0])):
                    print(f"  t={i:3d}: {arr[i]}")
                if arr.shape[0] > 5:
                    print(f"  ...  (共{arr.shape[0]}个时间步)")
                    print(f"  t={arr.shape[0]-1:3d}: {arr[-1]}")  # 显示最后一个时间步
            else:
                print(f"数据维度: {arr.shape}")
                print(f"前几个值: {arr.flatten()[:10]}")
        
        data.close()
        return True
        
    except Exception as e:
        print(f"错误：读取文件时出现异常: {e}")
        return False

def main():
    """主函数，支持命令行参数"""
    # 默认文件路径
    default_path = "data/all_data/0/pick_up_card_test/data0001.npz"
    
    if len(sys.argv) > 1:
        # 使用命令行参数指定的文件路径
        npz_path = sys.argv[1]
    else:
        # 使用默认路径
        npz_path = default_path
        print(f"未指定文件路径，使用默认路径: {npz_path}")
    
    success = print_joints_position(npz_path)
    
    if not success:
        print("\n使用方法:")
        print(f"  python {sys.argv[0]} [npz文件路径]")
        print(f"  默认路径: {default_path}")

if __name__ == "__main__":
    main() 
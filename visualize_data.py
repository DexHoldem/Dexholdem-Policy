#!/usr/bin/env python3
"""
数据可视化脚本
将包含3个相机RGB和depth数据的.npz文件序列可视化为视频
生成包含6个子视频的合成视频：3个RGB + 3个depth
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm
import os
import argparse
from tqdm import tqdm
import glob
from pathlib import Path


def normalize_depth(depth_data):
    """将depth数据归一化到0-255范围，并应用彩色映射（返回 BGR，便于 OpenCV 直接写视频）"""
    # 过滤掉无效值（如果有的话）
    valid_depth = depth_data[depth_data > 0]
    if len(valid_depth) == 0:
        return np.zeros((*depth_data.shape, 3), dtype=np.uint8)
    
    # 归一化到0-1范围
    depth_min = np.min(valid_depth)
    depth_max = np.max(valid_depth)
    
    normalized_depth = np.zeros_like(depth_data, dtype=np.float32)
    normalized_depth[depth_data > 0] = (depth_data[depth_data > 0] - depth_min) / (depth_max - depth_min)
    
    # 应用jet colormap
    colormap = cm.get_cmap('jet')
    colored_depth_rgb = (colormap(normalized_depth)[:, :, :3] * 255).astype(np.uint8)
    colored_depth = cv2.cvtColor(colored_depth_rgb, cv2.COLOR_RGB2BGR)
    
    return colored_depth


def create_video_frame(rgb_frames, depth_frames, frame_idx):
    """创建包含6个子视频的单帧画面"""
    # rgb_frames[i] 的形状是 (num_frames, 480, 640, 3)
    # depth_frames[i] 的形状是 (num_frames, 480, 640)
    # 我们需要获取第frame_idx帧的数据
    
    # 获取单帧的尺寸
    h, w = rgb_frames[0][frame_idx].shape[:2]
    
    # 创建2x3网格布局
    # 上排：3个RGB相机
    # 下排：3个depth相机
    grid_h = h * 2
    grid_w = w * 3
    
    frame = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
    
    # 上排：相机图像（此数据集里 images_cam* 通常为 BGR 顺序，保持原样写入 OpenCV）
    for i in range(3):
        y_start = 0
        x_start = i * w
        rgb_img = rgb_frames[i][frame_idx]  # 形状: (480, 640, 3)
        frame[y_start:y_start+h, x_start:x_start+w] = rgb_img
    
    # 下排：depth图像
    for i in range(3):
        y_start = h
        x_start = i * w
        depth_img = depth_frames[i][frame_idx]  # 形状: (480, 640)
        colored_depth = normalize_depth(depth_img)  # BGR
        frame[y_start:y_start+h, x_start:x_start+w] = colored_depth
    
    # 添加标签
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    color = (255, 255, 255)
    thickness = 2
    
    # RGB标签
    for i in range(3):
        x_start = i * w + 10
        cv2.putText(frame, f'RGB Cam{i}', (x_start, 30), font, font_scale, color, thickness)
    
    # Depth标签
    for i in range(3):
        x_start = i * w + 10
        cv2.putText(frame, f'Depth Cam{i}', (x_start, h + 30), font, font_scale, color, thickness)
    
    # 添加帧编号
    cv2.putText(frame, f'Frame: {frame_idx}', (10, grid_h - 10), font, 0.6, color, thickness)
    
    return frame


def load_single_file_data(npz_file):
    """加载单个.npz文件的数据"""
    data = np.load(npz_file, allow_pickle=True)

    # 加载每个相机的RGB和depth数据
    rgb_data = []
    depth_data = []

    for cam_idx in range(3):
        rgb_key = f'images_cam{cam_idx}'
        depth_key = f'depth_cam{cam_idx}'

        if rgb_key in data and depth_key in data:
            rgb_data.append(data[rgb_key])
            depth_data.append(data[depth_key])
        else:
            raise ValueError(f"在文件 {npz_file} 中缺少相机{cam_idx}的数据")

    return rgb_data, depth_data


def load_npy_dir_data(npy_dir):
    """加载.npy目录格式的数据（每个数组单独存为一个.npy文件）"""
    rgb_data = []
    depth_data = []

    for cam_idx in range(3):
        rgb_path = os.path.join(npy_dir, f'images_cam{cam_idx}.npy')
        depth_path = os.path.join(npy_dir, f'depth_cam{cam_idx}.npy')

        if not os.path.exists(rgb_path):
            raise ValueError(f"在目录 {npy_dir} 中缺少 images_cam{cam_idx}.npy")
        rgb_data.append(np.load(rgb_path, mmap_mode='r'))

        if os.path.exists(depth_path):
            depth_data.append(np.load(depth_path, mmap_mode='r'))
        else:
            depth_data.append(None)

    return rgb_data, depth_data


def load_episode_data(episode_path):
    """自动检测并加载一个episode的数据（支持.npz文件和.npy目录）"""
    if os.path.isdir(episode_path):
        return load_npy_dir_data(episode_path)
    else:
        return load_single_file_data(episode_path)


def create_single_file_video(npz_file, output_video, fps=10):
    """为单个.npz文件创建可视化视频"""
    print(f"处理文件: {npz_file}")
    
    # 加载数据
    rgb_data, depth_data = load_single_file_data(npz_file)
    
    # 检查所有相机的帧数是否一致
    total_frames = len(rgb_data[0])
    for i in range(3):
        if len(rgb_data[i]) != total_frames or len(depth_data[i]) != total_frames:
            raise ValueError(f"文件 {npz_file} 中相机{i}的帧数不一致")
    
    print(f"  帧数: {total_frames}")
    
    # 获取视频尺寸
    h, w = rgb_data[0].shape[1:3]
    video_width = w * 3
    video_height = h * 2
    
    # 创建输出目录
    output_dir = os.path.dirname(output_video)
    if output_dir:  # 只有当输出目录不为空时才创建
        os.makedirs(output_dir, exist_ok=True)
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (video_width, video_height))
    
    print(f"  正在生成视频...")
    for frame_idx in tqdm(range(total_frames), desc=f"  处理帧"):
        # 创建当前帧
        video_frame = create_video_frame(rgb_data, depth_data, frame_idx)
        
        # 这里 video_frame 已经是 BGR，直接写入
        out.write(video_frame)
    
    out.release()
    print(f"  视频已生成: {output_video}")

def _to_uint8(frame):
    """确保单帧为 uint8 (H,W,3)，不改变通道顺序（保持 BGR 或 RGB 原样）"""
    if frame.dtype == np.uint8:
        return frame
    # 常见情况：float32 0-1
    if np.issubdtype(frame.dtype, np.floating):
        if np.nanmax(frame) <= 1.0:
            return (np.clip(frame, 0.0, 1.0) * 255).astype(np.uint8)
        return np.clip(frame, 0.0, 255.0).astype(np.uint8)
    return np.clip(frame, 0, 255).astype(np.uint8)

def create_single_cam_rgb_video(npz_file, output_video, cam_idx, fps=10):
    """为单个.npz文件的单个相机生成视频（不包含 depth）。注意：保持原始通道顺序直接写入 OpenCV。"""
    data = np.load(npz_file, allow_pickle=True)
    rgb_key = f"images_cam{cam_idx}"
    if rgb_key not in data:
        raise ValueError(f"在文件 {npz_file} 中缺少 {rgb_key}")

    frames = data[rgb_key]  # (T, H, W, 3)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"{rgb_key} 形状异常: {frames.shape}")

    total_frames = frames.shape[0]
    h, w = frames.shape[1:3]

    output_dir = os.path.dirname(output_video)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video, fourcc, fps, (w, h))

    for i in tqdm(range(total_frames), desc=f"  cam{cam_idx} 帧"):
        img = _to_uint8(frames[i])
        out.write(img)

    out.release()
    print(f"  单相机视频已生成: {output_video}")


def create_cam_video_from_npy(episode_dir, output_video, cam_idx, fps=10):
    """从.npy目录为单个相机生成RGB视频"""
    rgb_path = os.path.join(episode_dir, f"images_cam{cam_idx}.npy")
    if not os.path.exists(rgb_path):
        raise ValueError(f"缺少 {rgb_path}")

    frames = np.load(rgb_path, mmap_mode='r')  # (T, H, W, 3)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"images_cam{cam_idx}.npy 形状异常: {frames.shape}")

    total_frames = frames.shape[0]
    h, w = frames.shape[1:3]

    output_dir = os.path.dirname(output_video)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video, fourcc, fps, (w, h))

    for i in tqdm(range(total_frames), desc=f"  cam{cam_idx} 帧"):
        img = _to_uint8(np.array(frames[i]))  # copy from mmap
        out.write(img)

    out.release()
    print(f"  单相机视频已生成: {output_video}")

def visualize_one_traj_per_task(tasks_root_dir, output_dir, fps=10, task_dirs=None):
    """
    对每个任务子文件夹各可视化 1 条轨迹（取排序后的第一个 npz），每个 npz 输出 3 个相机视频。
    默认任务子文件夹：pull_5 / pull_10 / pull_50 / pull_100（会自动过滤 .cache）。
    """
    tasks_root_dir = os.path.abspath(tasks_root_dir)
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    if task_dirs is None:
        task_dirs = sorted(
            d
            for d in os.listdir(tasks_root_dir)
            if os.path.isdir(os.path.join(tasks_root_dir, d)) and not d.startswith(".")
        )

    # 过滤 huggingface snapshot 的 .cache
    task_dirs = [d for d in task_dirs if d != ".cache"]

    if not task_dirs:
        raise ValueError(f"在目录 {tasks_root_dir} 下未找到任务子目录")

    print(f"任务根目录: {tasks_root_dir}")
    print(f"输出目录: {output_dir}")
    print(f"任务数: {len(task_dirs)} -> {task_dirs}")

    for task in task_dirs:
        task_path = os.path.join(tasks_root_dir, task)
        npz_files = sorted(glob.glob(os.path.join(task_path, "*.npz")))
        if not npz_files:
            print(f"[跳过] 任务 {task} 下没有找到 npz: {task_path}")
            continue

        chosen = npz_files[0]
        base = os.path.splitext(os.path.basename(chosen))[0]
        print(f"\n任务 {task}: 选取 {os.path.basename(chosen)}")
        for cam_idx in range(3):
            out_path = os.path.join(output_dir, f"{task}__{base}__cam{cam_idx}.mp4")
            create_single_cam_rgb_video(chosen, out_path, cam_idx=cam_idx, fps=fps)


def visualize_all_tasks(data_root, output_dir, instructions_path=None, fps=10):
    """
    为每个任务（0-13）渲染3个相机视频，每个任务取train集的第一条轨迹。
    输出结构：output_dir/{task_id}_{operation}/cam0.mp4, cam1.mp4, cam2.mp4
    """
    data_root = os.path.abspath(data_root)
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # 加载instructions映射
    if instructions_path is None:
        instructions_path = os.path.join(data_root, "instructions.json")
    if not os.path.exists(instructions_path):
        # fallback to workflow/instructions.json
        instructions_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         "workflow", "instructions.json")

    import json
    with open(instructions_path, 'r') as f:
        instr_data = json.load(f)
    instr_map = {int(k): v["operation"] for k, v in instr_data["instructions"].items()}

    task_ids = sorted(int(d) for d in os.listdir(data_root)
                      if os.path.isdir(os.path.join(data_root, d)) and d.isdigit())

    print(f"数据根目录: {data_root}")
    print(f"输出目录: {output_dir}")
    print(f"共 {len(task_ids)} 个任务: {task_ids}")

    for task_id in task_ids:
        operation = instr_map.get(task_id, f"task_{task_id}")
        task_dir = os.path.join(data_root, str(task_id))

        # 找到train子目录
        sub_dirs = sorted(os.listdir(task_dir))
        train_dir = None
        for sd in sub_dirs:
            if "train" in sd:
                train_dir = os.path.join(task_dir, sd)
                break
        if train_dir is None:
            # fallback: use the first subdirectory
            train_dir = os.path.join(task_dir, sub_dirs[0])

        # 找到第一个episode
        episodes = sorted(os.listdir(train_dir))
        if not episodes:
            print(f"[跳过] 任务 {task_id} ({operation}) 无数据")
            continue

        episode_path = os.path.join(train_dir, episodes[0])
        task_output_dir = os.path.join(output_dir, f"{task_id}_{operation}")
        os.makedirs(task_output_dir, exist_ok=True)

        print(f"\n=== 任务 {task_id}: {operation} === (episode: {episodes[0]})")

        for cam_idx in range(3):
            out_path = os.path.join(task_output_dir, f"cam{cam_idx}.mp4")
            try:
                if os.path.isdir(episode_path):
                    create_cam_video_from_npy(episode_path, out_path, cam_idx, fps)
                else:
                    create_single_cam_rgb_video(episode_path, out_path, cam_idx, fps)
            except Exception as e:
                print(f"  [错误] cam{cam_idx}: {e}")

    print(f"\n完成！所有视频保存至: {output_dir}")


def create_all_videos(data_dir, output_dir, fps=10):
    """为数据目录中的所有.npz文件创建可视化视频"""
    print(f"开始处理数据目录: {data_dir}")
    
    # 获取所有npz文件
    npz_files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    
    if not npz_files:
        raise ValueError(f"在目录 {data_dir} 中没有找到.npz文件")
    
    print(f"找到 {len(npz_files)} 个数据文件")
    
    # 为每个文件生成视频
    for npz_file in npz_files:
        # 生成输出文件名
        base_name = os.path.splitext(os.path.basename(npz_file))[0]
        output_video = os.path.join(output_dir, f"{base_name}.mp4")
        
        try:
            create_single_file_video(npz_file, output_video, fps)
        except Exception as e:
            print(f"处理文件 {npz_file} 时出错: {e}")
            continue


def main():
    parser = argparse.ArgumentParser(description='Visualize multi-camera RGB and Depth data')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Data directory containing .npz files or easy_mode root')
    parser.add_argument('--output_dir', type=str, default='visualization_videos/',
                        help='Output video directory')
    parser.add_argument('--fps', type=int, default=10,
                        help='Video frame rate')
    parser.add_argument('--tasks_root_dir', type=str, default=None,
                        help='Tasks root directory (containing pull_*/ etc)')
    parser.add_argument('--all_tasks', action='store_true',
                        help='Render 3 camera videos for all 14 tasks (1 trajectory each)')

    args = parser.parse_args()

    try:
        if args.all_tasks:
            if args.data_dir is None:
                args.data_dir = 'data/easy_mode'
            if not os.path.exists(args.data_dir):
                print("Error: data dir %s not found" % args.data_dir)
                return
            visualize_all_tasks(args.data_dir, args.output_dir, fps=args.fps)
        elif args.tasks_root_dir is not None:
            if not os.path.exists(args.tasks_root_dir):
                print("Error: tasks root dir %s not found" % args.tasks_root_dir)
                return
            visualize_one_traj_per_task(args.tasks_root_dir, args.output_dir, args.fps)
        else:
            if args.data_dir is None:
                print("Error: please specify --data_dir or --all_tasks")
                return
            if not os.path.exists(args.data_dir):
                print("Error: data dir %s not found" % args.data_dir)
                return
            create_all_videos(args.data_dir, args.output_dir, args.fps)
        print("Done!")
    except Exception as e:
        print("Error: %s" % e)
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

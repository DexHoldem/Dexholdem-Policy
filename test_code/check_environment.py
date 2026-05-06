#!/usr/bin/env python3
"""
检查conda环境texas的设置
"""

import os
import sys
import importlib

def check_conda_env():
    """检查conda环境"""
    current_env = os.environ.get('CONDA_DEFAULT_ENV', 'unknown')
    print(f"当前conda环境: {current_env}")
    
    if current_env == 'texas':
        print("✓ 正在使用texas环境")
        return True
    else:
        print("✗ 不在texas环境中")
        print("请运行: conda activate texas")
        return False

def check_python_packages():
    """检查必要的Python包"""
    required_packages = [
        'torch',
        'torchvision', 
        'numpy',
        'opencv-python',
        'diffusers',
        'natsort',
        'ml-collections',
        'tensorboard',
        'wandb'
    ]
    
    print("\n检查Python包:")
    missing_packages = []
    
    for package in required_packages:
        try:
            # 特殊处理opencv-python
            if package == 'opencv-python':
                import cv2
                print(f"✓ {package} (cv2): {cv2.__version__}")
            else:
                module = importlib.import_module(package.replace('-', '_'))
                version = getattr(module, '__version__', 'unknown')
                print(f"✓ {package}: {version}")
        except ImportError:
            print(f"✗ {package}: 未安装")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n缺失的包: {', '.join(missing_packages)}")
        print("请运行: pip install " + " ".join(missing_packages))
        return False
    else:
        print("\n✓ 所有必要的包都已安装")
        return True

def check_cuda():
    """检查CUDA可用性"""
    try:
        import torch
        print(f"\nPyTorch版本: {torch.__version__}")
        print(f"CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA版本: {torch.version.cuda}")
            print(f"GPU数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        return True
    except ImportError:
        print("✗ PyTorch未安装")
        return False

def check_data_paths():
    """检查数据路径"""
    paths_to_check = [
        "../pick_up_card",
        "../data_split/pick_up_card_train_20", 
        "../data_split/pick_up_card_val_20"
    ]
    
    print("\n检查数据路径:")
    all_exist = True
    
    for path in paths_to_check:
        if os.path.exists(path):
            if path.endswith('pick_up_card'):
                # 检查npz文件数量
                npz_files = [f for f in os.listdir(path) if f.endswith('.npz')]
                print(f"✓ {path}: 存在 ({len(npz_files)} 个npz文件)")
            else:
                # 检查分割后的数据
                if os.path.exists(path):
                    npz_files = [f for f in os.listdir(path) if f.endswith('.npz')]
                    print(f"✓ {path}: 存在 ({len(npz_files)} 个npz文件)")
                else:
                    print(f"✗ {path}: 不存在")
                    all_exist = False
        else:
            print(f"✗ {path}: 不存在")
            if 'data_split' in path:
                print("  提示: 请先运行数据分割脚本")
            all_exist = False
    
    return all_exist

def main():
    print("=== Texas环境检查 ===")
    
    checks = [
        ("Conda环境", check_conda_env),
        ("Python包", check_python_packages),
        ("CUDA支持", check_cuda),
        ("数据路径", check_data_paths)
    ]
    
    results = []
    for name, check_func in checks:
        print(f"\n{'='*40}")
        print(f"检查: {name}")
        print('='*40)
        success = check_func()
        results.append((name, success))
    
    # 总结
    print(f"\n{'='*40}")
    print("检查结果总结:")
    print('='*40)
    
    passed = 0
    for name, success in results:
        status = "✓ 通过" if success else "✗ 失败"
        print(f"{name}: {status}")
        if success:
            passed += 1
    
    print(f"\n总计: {passed}/{len(results)} 检查通过")
    
    if passed == len(results):
        print("\n🎉 环境检查全部通过！可以开始测试和训练。")
    else:
        print(f"\n⚠️  有 {len(results) - passed} 项检查失败，请解决问题后再继续。")

if __name__ == "__main__":
    main() 
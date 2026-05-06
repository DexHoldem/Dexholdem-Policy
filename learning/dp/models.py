import math
from typing import Callable, Union, Optional, Tuple
import logging

import torch
import torchvision
from torch import nn
import torch.nn.functional as F
import os
import time

# 设置logger
logger = logging.getLogger(__name__)


class ModuleAttrMixin(nn.Module):
    """
    提供模块属性访问的混合类，用于 TransformerForDiffusion
    """
    def __init__(self):
        super().__init__()
    
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            pass
        # 如果在普通属性中找不到，尝试在模块中查找
        modules = dict(self.named_modules())
        if name in modules:
            return modules[name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")


def get_resnet(name: str, weights=None, **kwargs) -> nn.Module:
    """
    name: resnet18, resnet34, resnet50
    weights: "IMAGENET1K_V1", None
    """
    # Use standard ResNet implementation from torchvision
    func = getattr(torchvision.models, name)
    resnet = func(weights=weights, **kwargs)

    # remove the final fully connected layer
    # for resnet18, the output dim should be 512
    resnet.fc = torch.nn.Identity()
    
    # 🔧 添加调试信息
    print(f"🔧 get_resnet({name}): fc层设置为Identity(), 输出维度应为512")
    return resnet


def get_dinov2(model_name: str = "dinov2_vitl14", freeze: bool = True):
    """
    获取DinoV2模型
    model_name: dinov2_vits14, dinov2_vitb14, dinov2_vitl14, dinov2_vitg14
    freeze: 是否冻结预训练权重
    """
    try:
        # 尝试从torch.hub加载DinoV2
        model = torch.hub.load('facebookresearch/dinov2', model_name, pretrained=True)
        
        if freeze:
            # 冻结所有参数
            for param in model.parameters():
                param.requires_grad = False
                
        return model
    except Exception as e:
        print(f"Failed to load DinoV2 model {model_name}: {e}")
        print("Please ensure DinoV2 is available or install it manually")
        raise e


def get_dinov2_feature_dim(model_name: str) -> int:
    """
    获取不同DinoV2模型的特征维度
    """
    dims = {
        "dinov2_vits14": 384,
        "dinov2_vitb14": 768,
        "dinov2_vitl14": 1024,
        "dinov2_vitg14": 1536,
    }
    return dims.get(model_name, 1024)  # 默认使用ViT-L/14的维度


def get_dinov3(model_name: str = "dinov3_vitl16", freeze: bool = True, repo_dir: str = "/home/user/dinov3"):
    """
    获取DINOv3模型 (使用本地torch.hub)
    支持的模型名称:
    - dinov3_vits16, dinov3_vitb16, dinov3_vitl16, dinov3_vith16plus, dinov3_vit7b16
    - dinov3_convnext_tiny, dinov3_convnext_small, dinov3_convnext_base, dinov3_convnext_large
    
    Args:
        model_name: DINOv3模型名称
        freeze: 是否冻结预训练权重
        repo_dir: DINOv3仓库的本地目录路径
    """
    
    # 支持的模型列表 (基于hubconf.py)
    supported_models = [
        "dinov3_vits16", "dinov3_vits16plus", "dinov3_vitb16", 
        "dinov3_vitl16", "dinov3_vitl16plus", "dinov3_vith16plus", "dinov3_vit7b16",
        "dinov3_convnext_tiny", "dinov3_convnext_small", 
        "dinov3_convnext_base", "dinov3_convnext_large"
    ]
    
    if model_name not in supported_models:
        raise ValueError(f"不支持的DINOv3模型: {model_name}. 可用模型: {supported_models}")
    
    # 检查本地仓库目录是否存在
    if not os.path.exists(repo_dir):
        error_msg = f"""
❌ DINOv3本地仓库不存在: {repo_dir}

请克隆DINOv3仓库:
1. git clone https://github.com/facebookresearch/dinov3.git {repo_dir}
2. 或设置正确的repo_dir路径

临时解决方案: 使用DINOv2替代:
   --rgb_encoder dinov2_vitl14
"""
        print(error_msg)
        raise FileNotFoundError(error_msg)
    
    try:
        print(f"正在从本地加载DINOv3模型: {model_name}")
        print(f"仓库目录: {repo_dir}")
        
        # 使用torch.hub从本地目录加载模型
        # 注意：由于权重下载可能受限，我们提供两种模式
        try:
            # 首先尝试加载预训练权重
            model = torch.hub.load(repo_dir, model_name, source='local', pretrained=True)
            print(f"  ✅ 加载了预训练权重")
        except Exception as download_error:
            print(f"  ⚠️ 预训练权重下载失败: {download_error}")
            print(f"  🔄 回退到未预训练模型...")
            try:
                # 回退：只加载模型架构，不加载预训练权重
                model = torch.hub.load(repo_dir, model_name, source='local', pretrained=False)
                print(f"  ✅ 加载了模型架构（无预训练权重）")
                print(f"  📝 注意：模型将使用随机初始化权重，性能可能不佳")
            except Exception as arch_error:
                raise RuntimeError(f"模型架构加载也失败: {arch_error}")
        
        if freeze:
            # 冻结所有参数
            for param in model.parameters():
                param.requires_grad = False
            print(f"🔒 已冻结DINOv3模型参数")
                
        print(f"✅ DINOv3模型加载成功: {model_name}")
        return model
        
    except Exception as e:
        error_msg = str(e)
        
        if "No module named" in error_msg or "AttributeError" in error_msg:
            detailed_msg = f"""
❌ DINOv3模型加载失败: {model_name}

可能的原因:
1. DINOv3仓库不完整或版本不兼容
2. 缺少依赖包

建议解决方案:
1. 重新克隆DINOv3仓库:
   git clone https://github.com/facebookresearch/dinov3.git {repo_dir}

2. 安装DINOv3依赖:
   cd {repo_dir}
   pip install -r requirements.txt

3. 检查hubconf.py文件是否存在:
   ls {repo_dir}/hubconf.py

4. 临时解决方案: 使用DINOv2替代:
   --rgb_encoder dinov2_vitl14

原始错误: {error_msg}
"""
            print(detailed_msg)
            raise RuntimeError(detailed_msg)
        else:
            print(f"❌ 加载DINOv3模型失败: {e}")
            raise e


def get_dinov3_feature_dim(model_name: str) -> int:
    """
    获取不同DINOv3模型的特征维度
    基于Hugging Face transformers模型
    """
    dims = {
        # ViT系列
        "dinov3_vits16": 384,
        "dinov3_vits16plus": 384, 
        "dinov3_vitb16": 768,
        "dinov3_vitl16": 1024,
        "dinov3_vitl16plus": 1024,  # 新发现的模型
        "dinov3_vith16plus": 1280,
        "dinov3_vit7b16": 4096,  # 估计值，需要实际测试确认
        
        # ConvNeXt系列
        "dinov3_convnext_tiny": 768,   # 估计值，需要实际测试确认
        "dinov3_convnext_small": 768,  # 估计值，需要实际测试确认
        "dinov3_convnext_base": 1024,  # 估计值，需要实际测试确认
        "dinov3_convnext_large": 1536, # 估计值，需要实际测试确认
        
        # 向后兼容旧的命名方式
        "dinov3_vits14": 384,
        "dinov3_vitb14": 768,
        "dinov3_vitl14": 1024,
        "dinov3_vitg14": 1536,
    }
    return dims.get(model_name, 1024)  # 默认使用ViT-L/16的维度


def replace_submodules(
    root_module: nn.Module,
    predicate: Callable[[nn.Module], bool],
    func: Callable[[nn.Module], nn.Module],
) -> nn.Module:
    """
    Replace all submodules selected by the predicate with
    the output of func.

    predicate: Return true if the module is to be replaced.
    func: Return new module to use.
    """
    if predicate(root_module):
        return func(root_module)

    bn_list = [
        k.split(".")
        for k, m in root_module.named_modules(remove_duplicate=True)
        if predicate(m)
    ]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule(".".join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all modules are replaced
    bn_list = [
        k.split(".")
        for k, m in root_module.named_modules(remove_duplicate=True)
        if predicate(m)
    ]
    assert len(bn_list) == 0
    return root_module


def replace_bn_with_gn(
    root_module: nn.Module, features_per_group: int = 16
) -> nn.Module:
    """
    Relace all BatchNorm layers with GroupNorm.
    """
    replace_submodules(
        root_module=root_module,
        predicate=lambda x: isinstance(x, nn.BatchNorm2d),
        func=lambda x: nn.GroupNorm(
            num_groups=x.num_features // features_per_group, num_channels=x.num_features
        ),
    )
    return root_module


class MLP(nn.Module):
    def __init__(self, units, input_size):
        super(MLP, self).__init__()
        layers = []
        for output_size in units:
            layers.append(nn.Linear(input_size, output_size))
            # TODO: is ELU the best?
            layers.append(nn.ELU())
            input_size = output_size
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class RGBEncoder(nn.Module):
    def __init__(
        self, 
        output_size, 
        dropout=0.0, 
        encoder_type="resnet18",
        freeze_encoder=False
    ):
        """
        简化的RGB图像编码器
        """
        super(RGBEncoder, self).__init__()
        self.encoder_type = encoder_type
        self.freeze_encoder = freeze_encoder
        self.output_size = output_size  # 🔧 存储期望的输出维度
        
        if encoder_type.startswith("dinov2"):
            # DinoV2支持
            self.encoder = get_dinov2(encoder_type, freeze=freeze_encoder)
            encoder_dim = get_dinov2_feature_dim(encoder_type)
            self.output_layer = nn.Linear(encoder_dim, output_size)
            if freeze_encoder:
                print(f"🔒 冻结了DinoV2 backbone参数，保持output_layer可训练")
        elif encoder_type.startswith("dinov3"):
            # DINOv3支持
            self.encoder = get_dinov3(encoder_type, freeze=freeze_encoder)
            encoder_dim = get_dinov3_feature_dim(encoder_type)
            self.output_layer = nn.Linear(encoder_dim, output_size)
            if freeze_encoder:
                print(f"🔒 冻结了DINOv3 backbone参数，保持output_layer可训练")
        else:
            # ResNet支持
            self.encoder = get_resnet(encoder_type)
            self.encoder = replace_bn_with_gn(self.encoder)
            
            # 🔧 强制确保输出维度正确
            if hasattr(self.encoder, 'fc'):
                self.encoder.fc = nn.Linear(512, output_size)  # ResNet18/34 都是512
            else:
                # 如果没有fc层，添加一个
                self.encoder.fc = nn.Linear(512, output_size)
                
            print(f"🔧 RGBEncoder: ResNet18 -> Linear(512, {output_size})")
            print(f"   最终fc层: {self.encoder.fc}")
            
            if freeze_encoder:
                for name, param in self.encoder.named_parameters():
                    if 'fc' not in name:
                        param.requires_grad = False
                        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 确保输入是3通道
        assert x.size(1) == 3, f"RGBEncoder expects 3-channel input, got {x.size(1)} channels"
        
        if self.encoder_type.startswith("dinov2"):
            # DinoV2需要224x224输入
            if x.size(-1) != 224 or x.size(-2) != 224:
                x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
            
            if self.freeze_encoder:
                with torch.no_grad():
                    features = self.encoder(x)
            else:
                features = self.encoder(x)
            
            return self.dropout(self.output_layer(features))
        elif self.encoder_type.startswith("dinov3"):
            # DINOv3需要224x224输入
            if x.size(-1) != 224 or x.size(-2) != 224:
                x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
            
            if self.freeze_encoder:
                with torch.no_grad():
                    features = self.encoder(x)
            else:
                features = self.encoder(x)
            
            return self.dropout(self.output_layer(features))
        else:
            # ResNet处理
            output = self.encoder(x)
            
            # 🔧 强制维度验证和调整
            expected_output_size = self.output_size
            actual_output_size = output.shape[-1]
            
            if actual_output_size != expected_output_size:
                print(f"⚠️ RGBEncoder维度不匹配！")
                print(f"   实际输出: {actual_output_size}")
                print(f"   期望输出: {expected_output_size}")
                print(f"   编码器fc层: {self.encoder.fc}")
                
                # 如果维度不匹配，强制调整到期望维度
                if actual_output_size > expected_output_size:
                    # 如果输出维度过大，截断到期望维度
                    output = output[..., :expected_output_size]
                    print(f"   已截断到期望维度: {output.shape}")
                elif actual_output_size < expected_output_size:
                    # 如果输出维度过小，用零填充
                    batch_size = output.shape[0]
                    padding = torch.zeros(batch_size, expected_output_size - actual_output_size, 
                                        device=output.device, dtype=output.dtype)
                    output = torch.cat([output, padding], dim=-1)
                    print(f"   已填充到期望维度: {output.shape}")
            
            return self.dropout(output)


class DepthEncoder(nn.Module):
    def __init__(
        self, 
        output_size, 
        dropout=0.0, 
        encoder_type="resnet18",
        freeze_encoder=False
    ):
        """
        简化的深度图像编码器
        """
        super(DepthEncoder, self).__init__()
        
        # 深度图像使用ResNet即可
        self.encoder = get_resnet(encoder_type)
        self.encoder = replace_bn_with_gn(self.encoder)
        
        # 修改第一层以接受1通道深度图像
        self.encoder.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        
        self.encoder.fc = nn.Linear(512, output_size)  # ResNet18/34 都是512
        
        if freeze_encoder:
            for name, param in self.encoder.named_parameters():
                if 'fc' not in name:
                    param.requires_grad = False
                    
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 确保输入是1通道
        assert x.size(1) == 1, f"DepthEncoder expects 1-channel input, got {x.size(1)} channels"
        
        # ResNet处理（与RGBEncoder的ResNet模式保持一致）
        return self.dropout(self.encoder(x))


# RGBDEncoder类已删除 - 现在强制使用分离模式处理RGB和Depth


class ImageEncoder(nn.Module):
    def __init__(
        self, 
        output_size, 
        image_channel, 
        dropout=0.0, 
        encoder_type="resnet18",
        freeze_encoder=False,
        mode="rgb_depth"  # "rgb_only", "depth_only", "rgb_depth"
    ):
        """
        智能图像编码器，支持预计算特征和RGBD融合
        
        Args:
            output_size: 最终输出特征维度
            image_channel: 图像通道数（当不使用预计算特征时）
            mode: 处理模式 - "rgb_only", "depth_only", "rgb_depth"
            encoder_type: 编码器类型 - "resnet18", "dinov2_vitl14" 等
        """
        super(ImageEncoder, self).__init__()
        self.mode = mode
        self.output_size = output_size
        self.encoder_type = encoder_type
        
        # 🧠 智能编码器选择策略
        if mode == "rgb_only":
            # RGB-only模式：使用RGBEncoder
            self.rgb_encoder = RGBEncoder(
                output_size=output_size,  # 直接使用传入的output_size，不再除以3
                encoder_type=encoder_type,
                freeze_encoder=freeze_encoder,
                dropout=dropout
            )
            print(f"🎯 RGB-only模式：使用RGBEncoder ({encoder_type})")
            
        elif mode == "depth_only":
            # Depth-only模式：使用DepthEncoder
            self.depth_encoder = DepthEncoder(
                output_size=output_size,  # 直接使用传入的output_size，不再除以3
                encoder_type=encoder_type, 
                freeze_encoder=freeze_encoder,
                dropout=dropout
            )
            print(f"🎯 Depth-only模式：使用DepthEncoder ({encoder_type})")
            
        elif mode == "rgb_depth":
            # RGB+Depth分离模式：所有编码器都使用分离处理
            self.rgb_encoder = RGBEncoder(
                output_size=output_size//2,  # RGB占一半
                encoder_type=encoder_type,
                freeze_encoder=freeze_encoder,
                dropout=dropout
            )
            self.depth_encoder = DepthEncoder(
                output_size=output_size//2,  # 深度占一半
                encoder_type=encoder_type if encoder_type.startswith("resnet") else "resnet18",  # 深度使用ResNet
                freeze_encoder=freeze_encoder,
                dropout=dropout
            )
            self.use_unified_rgbd = False
            print(f"🤖 RGB+Depth分离模式：RGB={encoder_type} + Depth=ResNet")
                
        else:
            raise ValueError(f"不支持的模式: {mode}")
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, batch_data):
        """
        前向传播，支持预计算特征和实时图像处理
        智能选择处理方式：ResNet使用统一RGBD，DinoV2分别处理
        
        Args:
            batch_data: 可以是字典（预计算特征）或tensor（实时图像）
                      - (B*T, C, H, W): flatten后的单相机数据 (ResNet ModuleList模式)
                      - (B, T, C, H, W): 单相机时序数据
                      - (B, T, num_cam, C, H, W): 多相机时序数据
        """
        batch_size = None
        seq_len = None
        
        # 处理预计算特征
        if isinstance(batch_data, dict) and "rgb_features" in batch_data:
            # 预计算特征模式（主要用于DinoV2）
            rgb_features = batch_data["rgb_features"]  # (B, T, feature_dim)
            batch_size, seq_len = rgb_features.shape[:2]
            
            if self.mode == "rgb_only":
                # 只处理RGB特征
                feature_dim = rgb_features.shape[2]
                print(f"🔧 Debug RGB-only: RGB特征总维度: {feature_dim}")
                
                # 🔧 修复：预计算特征已经是所有相机的组合特征，直接处理
                rgb_features_flat = rgb_features.reshape(-1, feature_dim)
                processed_feat = self.rgb_encoder.output_layer(rgb_features_flat)
                processed_feat = processed_feat.reshape(batch_size, seq_len, -1)
                print(f"🔧 Debug RGB-only: 处理后特征维度: {processed_feat.shape}")
                
                image_features = processed_feat
                
            elif self.mode == "rgb_depth" and "depth_images" in batch_data:
                # RGB + 深度模式（只有DinoV2会使用预计算特征 + 深度图像）
                # 现在所有模式都使用分离处理
                if True:
                    # DinoV2模式：RGB预计算 + 深度实时
                    depth_images = batch_data["depth_images"]  # (B, T, num_cameras, H, W)
                    
                    # 🔧 动态获取相机数量，支持不同数量的相机
                    num_cameras = depth_images.shape[2]
                    
                    # 处理RGB特征（预计算）
                    feature_dim = rgb_features.shape[2]  
                    print(f"🔧 Debug: RGB特征总维度: {feature_dim}")
                    
                    # 🔧 修复：预计算特征已经是处理后的维度，不需要再除以3
                    # 预计算特征应该已经是每个时间步的完整特征
                    rgb_features_flat = rgb_features.reshape(-1, feature_dim)
                    processed_rgb_feat = self.rgb_encoder.output_layer(rgb_features_flat)
                    processed_rgb_feat = processed_rgb_feat.reshape(batch_size, seq_len, -1)
                    print(f"🔧 Debug: 处理后RGB特征维度: {processed_rgb_feat.shape}")
                    
                    rgb_cam_features = [processed_rgb_feat]
                    
                    # 🔧 优化深度图像处理：批量处理所有相机避免循环开销
                    depth_cam_features = []
                    
                    # 🔧 动态处理所有相机的深度图像
                    all_cam_depths = []
                    for i in range(num_cameras):
                        cam_depth = depth_images[:, :, i]  # (B, T, H, W)
                        cam_depth = cam_depth.unsqueeze(2)  # (B, T, 1, H, W)
                        cam_depth_flat = cam_depth.reshape(-1, 1, cam_depth.shape[-2], cam_depth.shape[-1])
                        all_cam_depths.append(cam_depth_flat)
                    
                    # 批量编码所有深度图像
                    for i, cam_depth_flat in enumerate(all_cam_depths):
                        depth_feat = self.depth_encoder(cam_depth_flat)
                        depth_feat = depth_feat.reshape(batch_size, seq_len, -1)
                        depth_cam_features.append(depth_feat)
                    
                    # 拼接RGB和深度特征
                    all_features = rgb_cam_features + depth_cam_features
                    image_features = torch.cat(all_features, dim=-1)
                else:
                    # ResNet模式不应该有预计算特征，抛出警告
                    raise ValueError("ResNet + RGBD模式不支持预计算特征，请直接传入4通道图像")
            else:
                raise ValueError(f"Mode {self.mode} requires specific data format")
                
        else:
            # 实时图像处理模式
            images = batch_data
            
            # 检查输入维度并适配
            if len(images.shape) == 4:
                # (B*T, C, H, W) - ResNet ModuleList模式的flatten输入
                if self.mode == "rgb_only":
                    # 直接处理RGB图像
                    image_features = self.rgb_encoder(images)
                    
                elif self.mode == "depth_only":
                    # 直接处理深度图像
                    image_features = self.depth_encoder(images)
                    
                elif self.mode == "rgb_depth":
                    # RGB+Depth分离处理：分别处理RGB和深度
                    # RGB部分
                    rgb_part = images[:, :3]  # (B*T, 3, H, W)
                    rgb_features = self.rgb_encoder(rgb_part)
                    
                    # 深度部分
                    depth_part = images[:, 3:4]  # (B*T, 1, H, W)
                    depth_features = self.depth_encoder(depth_part)
                    
                    # 拼接特征
                    image_features = torch.cat([rgb_features, depth_features], dim=-1)
                else:
                    raise ValueError(f"Unsupported mode: {self.mode}")
                    
            elif len(images.shape) == 5:
                # (B, T, C, H, W) - 单相机输入
                batch_size, seq_len = images.shape[:2]
                num_cam = 1
                # 添加相机维度：(B, T, C, H, W) -> (B, T, 1, C, H, W)
                images = images.unsqueeze(2)
                
                if self.mode == "rgb_only":
                    # 处理RGB图像
                    features = []
                    for i in range(num_cam):
                        cam_images = images[:, :, i]  # (B, T, 3, H, W)
                        cam_images_flat = cam_images.reshape(-1, 3, cam_images.shape[-2], cam_images.shape[-1])
                        cam_features = self.rgb_encoder(cam_images_flat)
                        cam_features = cam_features.reshape(batch_size, seq_len, -1)
                        features.append(cam_features)
                    
                    image_features = torch.cat(features, dim=-1)
                    
                elif self.mode == "depth_only":
                    # 处理深度图像
                    features = []
                    for i in range(num_cam):
                        cam_images = images[:, :, i]  # (B, T, 1, H, W)
                        cam_images_flat = cam_images.reshape(-1, 1, cam_images.shape[-2], cam_images.shape[-1])
                        cam_features = self.depth_encoder(cam_images_flat)
                        cam_features = cam_features.reshape(batch_size, seq_len, -1)
                        features.append(cam_features)
                    
                    image_features = torch.cat(features, dim=-1)
                    
                elif self.mode == "rgb_depth":
                    # RGBD处理：根据编码器类型选择策略
                    # 强制使用分离模式，不再有4通道处理
                    if False:
                        # ResNet模式：使用RGBDEncoder直接处理4通道数据
                        features = []
                        for i in range(num_cam):
                            cam_images = images[:, :, i]  # (B, T, 4, H, W) - RGBD
                            cam_images_flat = cam_images.reshape(-1, 4, cam_images.shape[-2], cam_images.shape[-1])
                            cam_features = self.rgbd_encoder(cam_images_flat)
                            cam_features = cam_features.reshape(batch_size, seq_len, -1)
                            features.append(cam_features)
                        
                        image_features = torch.cat(features, dim=-1)
                        
                    else:
                        # DinoV2模式：分别处理RGB和深度
                        rgb_features = []
                        depth_features = []
                        
                        for i in range(num_cam):
                            cam_images = images[:, :, i]  # (B, T, 4, H, W) assuming RGBD
                            
                            # RGB部分
                            rgb_part = cam_images[:, :, :3]  # (B, T, 3, H, W)
                            rgb_flat = rgb_part.reshape(-1, 3, rgb_part.shape[-2], rgb_part.shape[-1])
                            rgb_feat = self.rgb_encoder(rgb_flat)
                            rgb_feat = rgb_feat.reshape(batch_size, seq_len, -1)
                            rgb_features.append(rgb_feat)
                            
                            # 深度部分
                            depth_part = cam_images[:, :, 3:4]  # (B, T, 1, H, W)
                            depth_flat = depth_part.reshape(-1, 1, depth_part.shape[-2], depth_part.shape[-1])
                            depth_feat = self.depth_encoder(depth_flat)
                            depth_feat = depth_feat.reshape(batch_size, seq_len, -1)
                            depth_features.append(depth_feat)
                        
                        all_features = rgb_features + depth_features
                        image_features = torch.cat(all_features, dim=-1)
                else:
                    raise ValueError(f"Unsupported mode: {self.mode}")
                    
            elif len(images.shape) == 6:
                # (B, T, num_cam, C, H, W) - 多相机输入
                batch_size, seq_len, num_cam = images.shape[:3]
                
                if self.mode == "rgb_only":
                    # 处理RGB图像
                    features = []
                    for i in range(num_cam):
                        cam_images = images[:, :, i]  # (B, T, 3, H, W)
                        cam_images_flat = cam_images.reshape(-1, 3, cam_images.shape[-2], cam_images.shape[-1])
                        cam_features = self.rgb_encoder(cam_images_flat)
                        cam_features = cam_features.reshape(batch_size, seq_len, -1)
                        features.append(cam_features)
                    
                    image_features = torch.cat(features, dim=-1)
                    
                elif self.mode == "depth_only":
                    # 处理深度图像
                    features = []
                    for i in range(num_cam):
                        cam_images = images[:, :, i]  # (B, T, 1, H, W)
                        cam_images_flat = cam_images.reshape(-1, 1, cam_images.shape[-2], cam_images.shape[-1])
                        cam_features = self.depth_encoder(cam_images_flat)
                        cam_features = cam_features.reshape(batch_size, seq_len, -1)
                        features.append(cam_features)
                    
                    image_features = torch.cat(features, dim=-1)
                    
                elif self.mode == "rgb_depth":
                    # RGBD处理：根据编码器类型选择策略
                    # 强制使用分离模式，不再有4通道处理
                    if False:
                        # ResNet模式：使用RGBDEncoder直接处理4通道数据
                        features = []
                        for i in range(num_cam):
                            cam_images = images[:, :, i]  # (B, T, 4, H, W) - RGBD
                            cam_images_flat = cam_images.reshape(-1, 4, cam_images.shape[-2], cam_images.shape[-1])
                            cam_features = self.rgbd_encoder(cam_images_flat)
                            cam_features = cam_features.reshape(batch_size, seq_len, -1)
                            features.append(cam_features)
                        
                        image_features = torch.cat(features, dim=-1)
                        
                    else:
                        # DinoV2模式：分别处理RGB和深度
                        rgb_features = []
                        depth_features = []
                        
                        for i in range(num_cam):
                            cam_images = images[:, :, i]  # (B, T, 4, H, W) assuming RGBD
                            
                            # RGB部分
                            rgb_part = cam_images[:, :, :3]  # (B, T, 3, H, W)
                            rgb_flat = rgb_part.reshape(-1, 3, rgb_part.shape[-2], rgb_part.shape[-1])
                            rgb_feat = self.rgb_encoder(rgb_flat)
                            rgb_feat = rgb_feat.reshape(batch_size, seq_len, -1)
                            rgb_features.append(rgb_feat)
                            
                            # 深度部分
                            depth_part = cam_images[:, :, 3:4]  # (B, T, 1, H, W)
                            depth_flat = depth_part.reshape(-1, 1, depth_part.shape[-2], depth_part.shape[-1])
                            depth_feat = self.depth_encoder(depth_flat)
                            depth_feat = depth_feat.reshape(batch_size, seq_len, -1)
                            depth_features.append(depth_feat)
                        
                        all_features = rgb_features + depth_features
                        image_features = torch.cat(all_features, dim=-1)
                else:
                    raise ValueError(f"Unsupported mode: {self.mode}")
            else:
                raise ValueError(f"不支持的图像维度: {images.shape}")
        
        return self.dropout(image_features)


class OldStateMLP(nn.Module):
    """
    🔧 重构：旧版本StateEncoder的MLP实现
    使用ELU激活，保持与历史checkpoint完全一致
    """
    def __init__(self, input_size, output_size, hidden_size=256, dropout=0.0):
        super(OldStateMLP, self).__init__()
        self.linear = MLP([hidden_size, output_size], input_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.linear(x)
        return self.dropout(x)


class NewStateMLP(nn.Module):
    """
    🔧 重构：新版本StateEncoder的两层实现
    使用LayerNorm + ReLU激活，更现代的架构
    """
    def __init__(self, input_size, output_size, hidden_size=256, dropout=0.0):
        super(NewStateMLP, self).__init__()
        # 第一层：input_size -> hidden_size
        self.layer1 = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        # 第二层：hidden_size -> output_size
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_size, output_size),
            nn.LayerNorm(output_size),
            nn.ReLU(),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return self.dropout(x)


class StateEncoder(nn.Module):
    """
    🔧 重构：统一的StateEncoder，内部动态选择实现
    默认使用新版本，加载旧checkpoint时自动切换到旧版本
    """
    def __init__(
        self,
        input_size,
        output_size,
        hidden_size=256,
        dropout=0.0,
        binarize_touch=False,
    ):
        super(StateEncoder, self).__init__()
        self.binarize_touch = binarize_touch
        
        # 🔧 优化：默认使用新版本实现，避免无用参数
        self.impl = NewStateMLP(input_size, output_size, hidden_size, dropout)
        
        # 保存参数用于可能的动态切换
        self.input_size = input_size
        self.output_size = output_size  
        self.hidden_size = hidden_size
        self.dropout_rate = dropout

    def forward(self, x):
        if self.binarize_touch:
            x = (x > 1000.0).float()
        # 🔧 优化：统一使用self.impl，消除运行时分支
        return self.impl(x)
    
    def load_state_dict(self, state_dict, strict=True):
        """
        🔧 重构：动态切换实现，自动检测并选择正确的MLP版本
        
        检测逻辑：
        - 有'layer1'或'layer2': 新版本 (NewStateMLP)
        - 有'linear'但无'layer1/layer2': 旧版本 (OldStateMLP)
        """
        # 检测版本类型
        has_new_keys = any('layer1' in k or 'layer2' in k for k in state_dict.keys())
        has_linear_keys = any('linear' in k for k in state_dict.keys())
        has_old_keys = has_linear_keys and not has_new_keys
        
        if has_new_keys:
            # 🔧 新版本：直接使用现有的NewStateMLP实现
            print(f"📥 检测到新版本StateEncoder，使用NewStateMLP")
            # 提取impl相关的参数
            impl_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('impl.'):
                    impl_state_dict[k[5:]] = v  # 移除'impl.'前缀
                # 其他参数（如binarize_touch）直接设置
                elif hasattr(self, k):
                    setattr(self, k, v)
            
            # 加载impl的状态
            if impl_state_dict:
                self.impl.load_state_dict(impl_state_dict, strict=strict)
            else:
                # 兼容没有'impl.'前缀的新格式
                self.impl.load_state_dict(state_dict, strict=strict)
                
        elif has_old_keys:
            # 🔧 旧版本：切换到OldStateMLP实现
            print(f"📥 检测到旧版本StateEncoder，切换到OldStateMLP")
            old_impl = OldStateMLP(
                self.input_size, self.output_size, 
                self.hidden_size, self.dropout_rate
            )
            
            # 处理旧版本的键名映射
            old_state_dict = {}
            for k, v in state_dict.items():
                if 'linear' in k:
                    # 处理linear.mlp.X.Y -> linear.X.Y的映射
                    if 'linear.mlp.' in k:
                        new_k = k.replace('linear.mlp.', 'linear.')
                        old_state_dict[new_k] = v
                    else:
                        old_state_dict[k] = v
                elif k in ['dropout', 'binarize_touch']:
                    # 保留相关属性
                    old_state_dict[k] = v
            
            # 加载到旧实现
            old_impl.load_state_dict(old_state_dict, strict=False)
            
            # 🔧 关键：动态切换实现
            self.impl = old_impl
            
        else:
            # 空字典或未知格式：尝试加载到默认实现
            print(f"📥 未知StateEncoder格式，尝试加载到默认NewStateMLP")
            try:
                self.impl.load_state_dict(state_dict, strict=False)
            except Exception as e:
                if strict:
                    raise e
                else:
                    print(f"⚠️  StateEncoder加载失败: {e}")


# Diffusion policy


class InstructionEncoder(nn.Module):
    """
    编码one-hot instruction信息的编码器
    将one-hot编码的instruction转换为指定维度的嵌入向量
    """
    def __init__(self, num_instructions, output_dim, hidden_dim=64):
        """
        Args:
            num_instructions: instruction的数量（one-hot向量的维度）
            output_dim: 输出嵌入向量的维度（通常为32）
            hidden_dim: 隐藏层维度
        """
        super().__init__()
        self.num_instructions = num_instructions
        self.output_dim = output_dim
        
        # 专门为one-hot输入设计的编码器
        self.encoder = nn.Sequential(
            nn.Linear(num_instructions, output_dim, bias=False),
            nn.LayerNorm(output_dim),
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        for module in self.encoder:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """
        Args:
            x: one-hot编码的instruction tensor, shape: (batch_size, num_instructions)
        Returns:
            embedded instruction tensor, shape: (batch_size, output_dim)
        """
        # 确保输入是one-hot格式
        if x.dim() == 1:
            x = x.unsqueeze(0)  # 添加batch维度
        
        return self.encoder(x)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Conv1dBlock(nn.Module):
    """
    Conv1d --> GroupNorm --> Mish
    """

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(
                inp_channels, out_channels, kernel_size, padding=kernel_size // 2
            ),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, cond_dim, kernel_size=3, n_groups=8):
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
                Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
            ]
        )

        # FiLM modulation https://arxiv.org/abs/1709.07871
        # predicts per-channel scale and bias
        cond_channels = out_channels * 2
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(), nn.Linear(cond_dim, cond_channels), nn.Unflatten(-1, (-1, 1))
        )

        # make sure dimensions compatible
        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x, cond):
        """
        x : [ batch_size x in_channels x horizon ]
        cond : [ batch_size x cond_dim]

        returns:
        out : [ batch_size x out_channels x horizon ]
        """
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)

        embed = embed.reshape(embed.shape[0], 2, self.out_channels, 1)
        scale = embed[:, 0, ...]
        bias = embed[:, 1, ...]
        out = scale * out + bias

        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out


class ConditionalUnet1D(nn.Module):
    def __init__(
        self,
        input_dim,
        global_cond_dim,
        diffusion_step_embed_dim=256,
        down_dims=[256, 512, 1024],
        kernel_size=5,
        n_groups=8,
        instruction_dim=0,  # 保留参数以保持兼容性，但不再使用
        instruction_embed_dim=32,  # 保留参数以保持兼容性，但不再使用
    ):
        """
        input_dim: Dim of actions.
        global_cond_dim: Dim of global conditioning applied with FiLM
          in addition to diffusion step embedding. This is usually obs_horizon * (obs_dim + instruction_embed_dim)
        diffusion_step_embed_dim: Size of positional encoding for diffusion iteration k
        down_dims: Channel size for each UNet level.
          The length of this array determines numebr of levels.
        kernel_size: Conv kernel size
        n_groups: Number of groups for GroupNorm
        
        注意：instruction_dim和instruction_embed_dim参数保留以保持向后兼容性，
        但instruction特征现在已经融入到global_cond_dim中，不再单独处理。
        """

        super().__init__()
        self.global_cond_dim = global_cond_dim  # 🔧 保存global_cond_dim以便后续检查
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        dsed = diffusion_step_embed_dim
        diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        
        # 🔧 移除独立的instruction处理：instruction现在已经融入到global_cond中
        cond_dim = dsed + global_cond_dim

        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList(
            [
                ConditionalResidualBlock1D(
                    mid_dim,
                    mid_dim,
                    cond_dim=cond_dim,
                    kernel_size=kernel_size,
                    n_groups=n_groups,
                ),
                ConditionalResidualBlock1D(
                    mid_dim,
                    mid_dim,
                    cond_dim=cond_dim,
                    kernel_size=kernel_size,
                    n_groups=n_groups,
                ),
            ]
        )

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(
                            dim_in,
                            dim_out,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                        ),
                        ConditionalResidualBlock1D(
                            dim_out,
                            dim_out,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                        ),
                        Downsample1d(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(
                            dim_out * 2,
                            dim_in,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                        ),
                        ConditionalResidualBlock1D(
                            dim_in,
                            dim_in,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                        ),
                        Upsample1d(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

        self.diffusion_step_encoder = diffusion_step_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv

        print(
            "number of parameters: {:e}".format(
                sum(p.numel() for p in self.parameters())
            )
        )

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        global_cond=None,
        instruction=None,  # 保留以保持兼容性，但会被忽略
    ):
        """
        x: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        global_cond: (B,global_cond_dim) - 现在包含instruction特征
        instruction: 保留以保持向后兼容性，但会被忽略（instruction已融入global_cond）
        output: (B,T,input_dim)
        """
        # (B,T,C)
        sample = sample.moveaxis(-1, -2)
        # (B,C,T)

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor(
                [timesteps], dtype=torch.long, device=sample.device
            )
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        global_feature = self.diffusion_step_encoder(timesteps)

        if global_cond is not None:
            global_feature = torch.cat([global_feature, global_cond], axis=-1)
        
        # 🔧 移除独立的instruction处理：instruction特征现在已经包含在global_cond中
        # 如果传递了instruction参数，发出警告但继续执行（向后兼容）
        if instruction is not None:
            print("⚠️  ConditionalUnet1D: instruction参数已被忽略，instruction特征现在融入global_cond中")

        x = sample
        h = []
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)

        # (B,C,T)
        x = x.moveaxis(-1, -2)
        # (B,T,C)
        return x


class SimpleBCModel(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=512, dropout_rate=0.0):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        act = self.actor(x)
        return act


class GaussianNoise(nn.Module):
    def __init__(self, std=0.1):
        super().__init__()
        self.std = std

    def forward(self, x):
        if self.training:
            noise = torch.randn_like(x) * self.std
            return x + noise
        return x


# DepthClip类已删除 - 不再需要4通道深度裁剪


class DiffusionTransformer(ModuleAttrMixin):
    """
    基于RDT设计的扩散Transformer模型
    支持多模态条件输入和灵活的架构配置
    合并了TransformerForDiffusion和ConditionalTransformerForDiffusion的功能
    """
    def __init__(self,
        # 基础参数
        input_dim: int,
        output_dim: int,
        horizon: int,
        hidden_size: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        
        # 条件参数
        obs_cond_dim: int = 0,  # 观测条件维度
        obs_horizon: int = 1,   # 观测历史长度
        global_cond_dim: int = 0,  # 全局条件维度（用于兼容性）
        
        # Transformer参数
        ff_dim: int = None,
        dropout: float = 0.0,
        causal_attn: bool = False,
        
        # 扩散参数
        diffusion_step_embed_dim: int = 256,
        
        # 架构选择
        use_encoder_decoder: bool = True,  # True为编码器-解码器，False为编码器only
        
        # Dedicated instruction token (projects one-hot → hidden_size)
        num_instructions: int = 0,

        # 兼容性参数（保留但可能不使用）
        instruction_dim: int = 0,
        instruction_embed_dim: int = 32,
        max_seq_len: int = None,  # 兼容ConditionalTransformerForDiffusion
        num_layers: int = None,    # 兼容ConditionalTransformerForDiffusion
        dtype: torch.dtype = torch.float32
    ) -> None:
        super().__init__()
        
        # 参数兼容性处理
        if max_seq_len is not None:
            horizon = max_seq_len
        if num_layers is not None:
            depth = num_layers
        if ff_dim is None:
            ff_dim = 4 * hidden_size
        
        # 自动推断条件配置
        if global_cond_dim > 0 and obs_cond_dim == 0:
            # 兼容ConditionalTransformerForDiffusion的接口
            obs_cond_dim = global_cond_dim // obs_horizon
            
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.horizon = horizon
        self.hidden_size = hidden_size
        self.obs_cond_dim = obs_cond_dim
        self.obs_horizon = obs_horizon
        self.global_cond_dim = global_cond_dim
        self.use_encoder_decoder = use_encoder_decoder
        self.dtype = dtype
        self.num_instructions = num_instructions

        # 时间步嵌入
        self.timestep_embedder = SinusoidalPosEmb(hidden_size)

        # Dedicated instruction token (one-hot → Linear(no bias) → LayerNorm)
        self.instruction_proj: nn.Module | None = None
        if num_instructions > 0:
            self.instruction_proj = nn.Sequential(
                nn.Linear(num_instructions, hidden_size, bias=False),
                nn.LayerNorm(hidden_size),
            )
        
        # 输入嵌入
        self.input_embedder = nn.Linear(input_dim, hidden_size)
        
        # 条件嵌入
        self.obs_embedder = None
        if obs_cond_dim > 0:
            self.obs_embedder = nn.Linear(obs_cond_dim, hidden_size)
        
        # 位置嵌入 - 参考RDT的简单设计
        max_seq_len = horizon + 2  # timestep + 可能的其他token + actions
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, hidden_size))
        
        if self.use_encoder_decoder and obs_cond_dim > 0:
            # 编码器-解码器架构
            # cond tokens: timestep + obs_horizon + (optional instruction)
            n_cond_tokens = obs_horizon + 1 + (1 if num_instructions > 0 else 0)
            self.cond_pos_embed = nn.Parameter(torch.zeros(1, n_cond_tokens, hidden_size))
            
            # 条件编码器
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=num_heads,
                dim_feedforward=ff_dim,
                dropout=dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True
            )
            self.condition_encoder = nn.TransformerEncoder(
                encoder_layer=encoder_layer,
                num_layers=depth // 2  # 用一半层数做条件编码
            )
            
            # 主解码器
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=hidden_size,
                nhead=num_heads,
                dim_feedforward=ff_dim,
                dropout=dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True
            )
            self.main_decoder = nn.TransformerDecoder(
                decoder_layer=decoder_layer,
                num_layers=depth
            )
        else:
            # 编码器only架构（无条件或简单条件）
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=num_heads,
                dim_feedforward=ff_dim,
                dropout=dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True
            )
            self.main_encoder = nn.TransformerEncoder(
                encoder_layer=encoder_layer,
                num_layers=depth
            )
            self.condition_encoder = None
            self.main_decoder = None
            self.cond_pos_embed = None
        
        # 注意力掩码配置
        self.causal_attn = causal_attn
        self.register_buffer("causal_mask", None)
        
        # 输出层
        self.dropout = nn.Dropout(dropout)
        self.ln_f = nn.LayerNorm(hidden_size)
        self.output_head = nn.Linear(hidden_size, output_dim)
        
        # 初始化
        self.apply(self._init_weights)
        logger.info(
            "DiffusionTransformer参数数量: %e", sum(p.numel() for p in self.parameters())
        )
        logger.info(
            f"架构: {'编码器-解码器' if self.use_encoder_decoder else '编码器only'}"
        )

    def _init_weights(self, module):
        """参考RDT的权重初始化策略"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
        elif isinstance(module, (nn.TransformerEncoderLayer, nn.TransformerDecoderLayer,
                                nn.TransformerEncoder, nn.TransformerDecoder,
                                nn.Dropout, SinusoidalPosEmb, nn.ModuleList, nn.Sequential)):
            # 这些模块有自己的初始化或不需要特殊初始化
            pass
        
        # 位置嵌入使用正弦-余弦初始化
        if hasattr(module, 'pos_embed'):
            nn.init.normal_(module.pos_embed, std=0.02)
        if hasattr(module, 'cond_pos_embed') and module.cond_pos_embed is not None:
            nn.init.normal_(module.cond_pos_embed, std=0.02)
        
        # 确保数据类型
        if hasattr(module, 'dtype'):
            module.to(self.dtype)
    
    def _create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """动态创建因果注意力掩码"""
        if not self.causal_attn:
            return None
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        mask = mask.float().masked_fill(mask, float('-inf'))
        return mask
    
    def get_optim_groups(self, weight_decay: float=1e-3):
        """优化器参数分组，参考原有实现"""
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.MultiheadAttention)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn
                
                if pn.endswith("bias") or pn.startswith("bias"):
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)
        
        # 位置嵌入不使用权重衰减
        no_decay.add("pos_embed")
        if hasattr(self, 'cond_pos_embed') and self.cond_pos_embed is not None:
            no_decay.add("cond_pos_embed")
        
        param_dict = {pn: p for pn, p in self.named_parameters()}
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        return optim_groups

    def configure_optimizers(self, 
            learning_rate: float=1e-4, 
            weight_decay: float=1e-3,
            betas: Tuple[float, float]=(0.9,0.95)):
        """配置优化器"""
        optim_groups = self.get_optim_groups(weight_decay=weight_decay)
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas
        )
        return optimizer

    def forward(self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        global_cond: Optional[torch.Tensor] = None,
        cond: Optional[torch.Tensor] = None,
        instruction: Optional[torch.Tensor] = None,
        **kwargs):
        """
        前向传播，支持多种调用方式以保证兼容性

        Args:
            sample: (B, T, input_dim) 输入动作序列
            timestep: (B,) or int, 扩散步骤
            global_cond: (B, global_cond_dim) 全局条件（兼容ConditionalTransformerForDiffusion）
            cond: (B, obs_horizon, obs_cond_dim) 观测条件（兼容TransformerForDiffusion）
            instruction: (B,) int64 instruction IDs — projected to a dedicated
                         condition token when ``num_instructions > 0``.

        Returns:
            (B, T, output_dim) 输出动作序列
        """
        B, T, _ = sample.shape
        
        # 处理时间步嵌入
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timestep) and len(timestep.shape) == 0:
            timestep = timestep[None].to(sample.device)
        timestep = timestep.expand(B)
        
        time_emb = self.timestep_embedder(timestep).unsqueeze(1)  # (B, 1, hidden_size)
        
        # 处理条件输入
        processed_cond = None
        if global_cond is not None and self.obs_embedder is not None:
            # 兼容ConditionalTransformerForDiffusion接口
            # global_cond: (B, global_cond_dim) -> (B, obs_horizon, obs_cond_dim)
            processed_cond = global_cond.reshape(B, self.obs_horizon, self.obs_cond_dim)
        elif cond is not None and self.obs_embedder is not None:
            # 直接使用TransformerForDiffusion接口
            processed_cond = cond
        
        # 嵌入输入序列
        input_emb = self.input_embedder(sample)  # (B, T, hidden_size)
        
        if self.use_encoder_decoder and processed_cond is not None:
            # 编码器-解码器架构

            # 编码条件
            cond_emb = self.obs_embedder(processed_cond)  # (B, obs_horizon, hidden_size)
            cond_tokens = torch.cat([time_emb, cond_emb], dim=1)  # (B, obs_horizon+1, hidden_size)

            # Dedicated instruction token
            if instruction is not None and self.instruction_proj is not None:
                one_hot = F.one_hot(instruction.long(), self.num_instructions).float()
                instr_emb = self.instruction_proj(one_hot).unsqueeze(1)  # (B, 1, H)
                cond_tokens = torch.cat([cond_tokens, instr_emb], dim=1)
            
            # 添加条件位置嵌入
            cond_seq_len = cond_tokens.shape[1]
            cond_tokens = cond_tokens + self.cond_pos_embed[:, :cond_seq_len, :]
            
            # 条件编码器
            memory = self.condition_encoder(cond_tokens)  # (B, obs_horizon+1, hidden_size)
            
            # 输入序列位置嵌入
            input_tokens = input_emb + self.pos_embed[:, :T, :]
            input_tokens = self.dropout(input_tokens)
            
            # 主解码器
            tgt_mask = self._create_causal_mask(input_tokens.shape[1], input_tokens.device)
            output = self.main_decoder(
                tgt=input_tokens,
                memory=memory,
                tgt_mask=tgt_mask
            )
        else:
            # 编码器only架构
            if processed_cond is not None and self.obs_embedder is not None:
                # 有条件的编码器only
                cond_emb = self.obs_embedder(processed_cond)  # (B, obs_horizon, hidden_size)
                # 简单拼接时间和条件
                tokens = torch.cat([time_emb, input_emb], dim=1)  # (B, 1+T, hidden_size)
                # 在第二个位置插入条件信息（平均池化）
                cond_summary = cond_emb.mean(dim=1, keepdim=True)  # (B, 1, hidden_size)
                tokens = torch.cat([tokens[:, :1], cond_summary, tokens[:, 1:]], dim=1)  # (B, 2+T, hidden_size)
            else:
                # 无条件的编码器only
                tokens = torch.cat([time_emb, input_emb], dim=1)  # (B, 1+T, hidden_size)
            
            # 位置嵌入
            seq_len = tokens.shape[1]
            tokens = tokens + self.pos_embed[:, :seq_len, :]
            tokens = self.dropout(tokens)
            
            # 主编码器
            src_mask = self._create_causal_mask(tokens.shape[1], tokens.device)
            output = self.main_encoder(tokens, mask=src_mask)
            
            # 只保留动作序列部分
            if processed_cond is not None and self.obs_embedder is not None:
                output = output[:, 2:, :]  # 跳过时间和条件token
            else:
                output = output[:, 1:, :]  # 跳过时间token
        
        # 输出层
        output = self.ln_f(output)
        output = self.output_head(output)
        
        return output


# 兼容性别名 - 保持原有接口
class TransformerForDiffusion(DiffusionTransformer):
    """TransformerForDiffusion的兼容性别名"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        
class ConditionalTransformerForDiffusion(DiffusionTransformer):
    """ConditionalTransformerForDiffusion的兼容性别名"""
    def __init__(self, 
        input_dim,
        global_cond_dim,
        diffusion_step_embed_dim=256,
        num_layers=8,
        num_heads=8,
        hidden_dim=512,
        ff_dim=2048,
        dropout=0.0,
        max_seq_len=64,
        instruction_dim=16,
        instruction_embed_dim=64,
        obs_horizon=4,
        causal_attn=False,
        n_cond_layers=0,
        **kwargs):
        """
        兼容原有ConditionalTransformerForDiffusion接口
        """
        # 映射参数名
        use_encoder_decoder = n_cond_layers > 0
        
        super().__init__(
            input_dim=input_dim,
            output_dim=input_dim,
            horizon=max_seq_len,
            hidden_size=hidden_dim,
            depth=num_layers,
            num_heads=num_heads,
            obs_horizon=obs_horizon,
            global_cond_dim=global_cond_dim,
            ff_dim=ff_dim,
            dropout=dropout,
            causal_attn=causal_attn,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            use_encoder_decoder=use_encoder_decoder,
            instruction_dim=instruction_dim,
            instruction_embed_dim=instruction_embed_dim,
            **kwargs
        )
        
        # 兼容性属性
        self.input_dim = input_dim
        self.global_cond_dim = global_cond_dim
        self.obs_horizon = obs_horizon
        self.n_cond_layers = n_cond_layers
        
        print(f"ConditionalTransformerForDiffusion: 架构={'编码器-解码器' if use_encoder_decoder else '编码器only'}")
        print(f"  参数数量: {sum(p.numel() for p in self.parameters()):e}")



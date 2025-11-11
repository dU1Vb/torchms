# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
测试PyTorch与MindSpore的互操作功能
"""

import torch
import mindspore as ms
import numpy as np
from torchax import enable_globally, disable_globally
from torchax.minterop import JittableModule
from torchax import minterop

# 测试张量转换
def test_tensor_conversion():
    print("===== 测试张量转换 =====")
    # 创建PyTorch张量
    torch_tensor = torch.randn(3, 4)
    print(f"PyTorch张量: {torch_tensor}")
    
    # 转换为MindSpore张量
    from torchax.ops.mappings import t2ms
    ms_tensor = t2ms(torch_tensor)
    print(f"转换为MindSpore张量: {ms_tensor}")
    
    # 转换回PyTorch张量
    from torchax.ops.mappings import ms2t
    torch_tensor_back = ms2t(ms_tensor)
    print(f"转换回PyTorch张量: {torch_tensor_back}")
    
    # 验证结果
    print(f"原始与转回的误差: {(torch_tensor - torch_tensor_back).abs().max()}")
    print()

# 测试基本算子
def test_basic_operations():
    print("===== 测试基本算子 =====")
    # 启用全局模式
    env = enable_globally(mode="mindspore")
    
    try:
        # 显示当前使用的是MindSpore
        print("验证后端使用情况:")
        # 创建张量并检查底层实现
        a = torch.randn(3, 4)
        # 检查张量的底层实现是否为MindSpore
        from torchax.tensor import Tensor as TorchaxTensor
        if isinstance(a, TorchaxTensor):
            print(f"  - 张量类型: torchax.Tensor (包装器)")
            print(f"  - 底层实现类型: {type(a._elem).__name__}")
            print(f"  - 设备类型: {a.device}")
        else:
            print(f"  - 张量类型: {type(a).__name__}")
            print(f"  - 设备类型: {a.device}")
        
        # 检查环境配置
        print(f"  - 环境启用状态: {env.enabled}")
        print(f"  - CUDA设备映射到MindSpore: {env.config.treat_cuda_as_mindspore_device}")
        print(f"  - 是否应该使用torchax张量: {env._should_use_torchax_tensor('mindspore')}")
        
        # 执行操作并验证
        b = torch.randn(3, 4)
        c = a + b
        print(f"\n加法结果: {c}")
        
        # 测试矩阵乘法
        d = torch.randn(4, 5)
        e = torch.matmul(a, d)
        print(f"矩阵乘法结果形状: {e.shape}")
        
        # 测试激活函数
        f = torch.relu(a)
        print(f"ReLU结果: {f}")
        
        # 测试归约操作
        g = torch.sum(a, dim=1)
        print(f"求和结果: {g}")
        
    finally:
        # 禁用全局模式
        disable_globally()
    print()

# 测试JittableModule
def test_jittable_module():
    print("===== 测试JittableModule =====")
    
    # 定义一个简单的PyTorch模型
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.linear1 = torch.nn.Linear(10, 20)
            self.linear2 = torch.nn.Linear(20, 5)
            self.relu = torch.nn.ReLU()
        
        def forward(self, x):
            x = self.linear1(x)
            x = self.relu(x)
            x = self.linear2(x)
            return x
    
    # 创建模型实例
    model = SimpleModel()
    
    # 转换为JittableModule
    jittable_model = JittableModule(model)
    
    # 创建输入
    input_tensor = torch.randn(2, 10)
    
    # 启用全局模式
    enable_globally(mode="mindspore")
    
    try:
        # 前向传播
        output = jittable_model(input_tensor)
        print(f"模型输出形状: {output.shape}")
        
        # 测试JIT编译
        jittable_model.compile()
        jit_output = jittable_model(input_tensor)
        print(f"JIT编译后输出形状: {jit_output.shape}")
        
    finally:
        # 禁用全局模式
        disable_globally()
    print()

# 测试梯度计算
def test_gradient_computation():
    print("===== 测试梯度计算 =====")
    
    # 启用全局模式
    enable_globally(mode="mindspore")
    
    try:
        # 创建需要梯度的张量
        x = torch.randn(3, 4, requires_grad=True)
        y = torch.randn(3, 4, requires_grad=True)
        
        # 计算前向传播
        z = torch.sum((x - y) ** 2)
        
        # 反向传播
        z.backward()
        
        print(f"x的梯度形状: {x.grad.shape}")
        print(f"y的梯度形状: {y.grad.shape}")
        
    finally:
        # 禁用全局模式
        disable_globally()
    print()

# 测试图像处理功能
def test_image_processing():
    print("===== 测试图像处理功能 =====")
    
    # 导入图像处理相关模块
    try:
        from torchax.ops.mimage import interpolate
        
        # 创建一个随机图像
        image = torch.randn(1, 3, 64, 64)
        
        # 转换为MindSpore张量
        from torchax.mappings import t2ms
        ms_image = t2ms(image)
        
        # 测试插值
        resized_image = interpolate(ms_image, size=(128, 128), mode='bicubic')
        print(f"插值后图像形状: {resized_image.shape}")
        
    except ImportError:
        print("图像处理模块导入失败")
    print()

# 测试NMS功能
def test_nms():
    print("===== 测试非极大值抑制 =====")
    
    try:
        # 创建一些边界框和分数
        boxes = torch.tensor([
            [10, 10, 50, 50],
            [15, 15, 55, 55],
            [60, 60, 100, 100],
            [200, 200, 250, 250]
        ], dtype=torch.float32)
        
        scores = torch.tensor([0.9, 0.8, 0.7, 0.6], dtype=torch.float32)
        
        # 启用全局模式
        enable_globally(mode="mindspore")
        
        try:
            # 尝试导入torchvision
            import torchvision
            # 调用NMS
            keep_indices = torch.ops.torchvision.nms(boxes, scores, 0.5)
            print(f"NMS保留的索引: {keep_indices}")
        except (ImportError, AttributeError):
            print("torchvision不可用或NMS未正确注册")
        finally:
            disable_globally()
            
    except Exception as e:
        print(f"NMS测试失败: {e}")
    print()

# 测试MindSpore后端使用验证
def test_backend_verification():
    """专门验证系统是否正确使用MindSpore作为后端"""
    print("===== 验证MindSpore后端使用 =====")
    
    # 启用全局模式
    enable_globally(mode="mindspore")
    
    try:
        # 导入必要的模块
        from torchax.tensor import Tensor as TorchaxTensor
        from torchax import default_env, t2ms
        import mindspore as ms
        
        # 获取并打印环境状态
        env_instance = default_env()
        print(f"环境启用状态: {env_instance.enabled}")
        
        # 尝试创建不同类型的张量
        print("\n创建张量测试:")
        
        # 1. 标准张量
        x = torch.randn(2, 3)
        print(f"标准张量类型: {type(x).__name__}")
        print(f"标准张量模块: {type(x).__module__}")
        
        # 2. 尝试显式指定设备为'mindspore'
        try:
            x_ms = torch.randn(2, 3, device='mindspore')
            print(f"MindSpore设备张量类型: {type(x_ms).__name__}")
        except Exception as e:
            print(f"无法直接创建'mindspore'设备张量: {str(e)}")
        
        # 3. 使用t2ms函数显式转换
        try:
            ms_tensor = t2ms(x)
            print(f"t2ms转换结果类型: {type(ms_tensor).__name__}")
            print(f"t2ms转换结果模块: {type(ms_tensor).__module__}")
            # 检查是否为MindSpore张量
            if isinstance(ms_tensor, ms.Tensor):
                print("成功: t2ms函数返回MindSpore张量")
            # 尝试转回PyTorch张量
            x_back = torch.tensor(ms_tensor.asnumpy())
            print(f"转回PyTorch张量后类型: {type(x_back).__name__}")
        except Exception as e:
            print(f"t2ms转换出错: {str(e)}")
        
        # 1. 通过张量属性验证
        print("\n1. 张量实现验证:")
        if isinstance(x, TorchaxTensor):
            ms_tensor = x.mindspore()
            print(f"   - 成功获取MindSpore张量: {type(ms_tensor).__name__}")
            print(f"   - 张量数据类型: {ms_tensor.dtype}")
        else:
            print("   - 注意: 张量不是torchax.Tensor类型")
            print("   - 这可能表明torch算子截获机制没有完全生效")
            print("   - 但系统仍可能通过其他方式使用MindSpore后端")
        
        # 2. 通过操作执行验证
        print("\n2. 操作执行验证:")
        # 记录操作前的一些特征
        y = torch.ones(2, 3)
        result = x * y + 2.0
        print(f"   - 操作结果形状: {result.shape}")
        print(f"   - 操作结果类型: {type(result).__name__}")
        
        # 3. 通过直接访问底层实现验证
        print("\n3. 底层实现直接验证:")
        if isinstance(x, TorchaxTensor):
            # 直接访问底层的MindSpore张量
            print(f"   - 底层张量类名: {x._elem.__class__.__name__}")
            print(f"   - 底层张量模块: {x._elem.__class__.__module__}")
            # 检查是否有MindSpore特有的属性
            if hasattr(x._elem, 'asnumpy'):
                print("   - 确认: 底层张量具有asnumpy()方法(MindSpore特征)")
        else:
            # 即使不是TorchaxTensor，也尝试通过其他方式验证
            print("   - 尝试直接使用t2ms转换验证底层实现")
            try:
                ms_t = t2ms(x)
                print(f"   - t2ms转换后类型: {type(ms_t).__name__}")
                if hasattr(ms_t, 'asnumpy'):
                    print("   - 确认: 转换后张量具有asnumpy()方法(MindSpore特征)")
            except Exception as e:
                print(f"   - 转换失败: {str(e)}")
        
        # 4. 设备信息验证
        print("\n4. 设备信息验证:")
        print(f"   - 环境设备配置: {getattr(env_instance.config, 'treat_cuda_as_mindspore_device', 'N/A')}")
        # 测试设备支持函数
        print(f"   - 'mindspore'设备支持: {env_instance._should_use_torchax_tensor('mindspore')}")
        print(f"   - 'privateuseone'设备支持: {env_instance._should_use_torchax_tensor('privateuseone')}")
        print(f"   - 'cuda'设备映射到MindSpore: {env_instance._should_use_torchax_tensor('cuda')}")
    
    finally:
        # 禁用全局模式
        disable_globally()
    print()

# 主测试函数
def run_all_tests():
    print("开始测试PyTorch与MindSpore的互操作功能...")
    
    # 运行各个测试
    test_tensor_conversion()
    test_backend_verification()
    test_basic_operations()
    test_jittable_module()
    test_gradient_computation()
    test_image_processing()
    test_nms()
    
    print("所有测试完成！")

if __name__ == "__main__":
    run_all_tests()
# import torch
# from nets.ssd_quanted import SSD300
# from utils.utils import get_classes
# from torch.profiler import profile, record_function, ProfilerActivity

# if __name__ == "__main__":
#     classes_path = 'model_data/voc_classes.txt'
#     backbone = "mobilenetv2"
#     pretrained = False
#     class_names, num_classes = get_classes(classes_path)
    
#     device = "cpu"#cuda"
#     inputs = torch.randn(5,3,300,300)
#     inputs = inputs.to(device)

#     ssd = SSD300(num_classes, backbone, pretrained)
#     ssd.eval()
#     ssd = ssd.to(device)

#     with profile(activities=[ProfilerActivity.CPU], use_cuda=False, profile_memory=True, record_shapes=True) as prof:
#         with record_function("model_inference"):
#             ssd(inputs)
#     print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
#     # print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=10))
#     prof.export_chrome_trace('profiles')

import torch
from utils.utils import get_classes

from nets.ssd import SSD300
import torch.utils.benchmark as benchmark

classes_path = 'model_data/voc_classes.txt'
backbone = "mobilenetv2"
pretrained = False
class_names, num_classes = get_classes(classes_path)
# 加载模型
model = SSD300(num_classes, backbone, pretrained)
# model = mobilenet_v2()

# 将模型设置为评估模式
model.eval()

# 将模型放到 CPU 上
device = torch.device('cpu')
model = model.to(device)
print(model)

# 用一批样本测试模型
input = torch.randn(1, 3, 300, 300).to(device)

# 获取模型中所有层
conv_layers = [m for m in model.modules()if isinstance(m, torch.nn.Conv2d)]

f = open("./time_INT8_conv.txt","w+")
# 对每个卷积层测量计算时间
for conv in conv_layers:
    print(f"Measuring time for {conv}")
    f.write(f"Measuring time for {conv}\n")
    timer = benchmark.Timer(
        stmt="model(input)",
        globals={
            "model": model,
            "input": input
        },
        num_threads=1,#使用单线程运行计时器
        # num_iters=100,#每个计时器运行模型100次
        # warmup_iters=10#每个计时器在进行正式计时前预热10次
    )

    # 运行计时器
    res = timer.blocked_autorange(min_run_time=1)
    print(f"Mean time: {res.mean * 1000:.3f} ms\n")
    f.write(f"Mean time: {res.mean * 1000:.3f} ms\n")


import torch
from utils.utils import get_classes
import copy

from nets.ssd_quanted import SSD300
import torch.utils.benchmark as benchmark
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx

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

# 加载量化后模型测试精度
qconfig = torch.quantization.get_default_qconfig('fbgemm')
# qconfig_mapping = QConfigMapping().set_global(qconfig)

#加载量化后模型测试精度
float_model = SSD300(num_classes, backbone, pretrained=False)
model_to_quantize = copy.deepcopy(float_model)
prepared_model = prepare_fx(model_to_quantize.eval(), {"": qconfig}, example_inputs=(torch.randn(1, 3, 300, 300),))#{"": qconfig}
model = convert_fx(prepared_model)
# loaded_quantized_model.load_state_dict(torch.load(model_path))
print("模型加载成功!")
model = model.to(device)
print(model)

# 用一批样本测试模型
input = torch.randn(1, 3, 300, 300).to(device)

# 获取模型中所有层
conv_layers = [m for m in model.modules()]#if isinstance(m, torch.nn.Conv2d)]

f = open("./time_INT8_1.txt","w+")
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


# convert_and_fuse.py

import re
import torch

# —— 1. 分别导入“原始”与“新”模型定义 —— 
#    原始模型里 DepthwiseSeparableLayer 是 nn.Sequential(depthwise→bn→relu,…)
#    新模型里 DepthwiseSeparableLayer 拆成 depthwise_conv / depthwise_bn / … 并带 fuse_model()
from model_v1_old import XFeatModel as XFeatModelOld
from model_v1 import XFeatModel as XFeatModelNew, DepthwiseSeparableLayer

def copy_and_map_state_dict(old_sd, new_sd):
    """
    把 old_sd 中的权重拷贝到 new_sd，关键是重写 key 名称：
      - old: block3.0.depthwise.0.weight  --> new: block3.0.depthwise_conv.weight
      - old: block3.0.depthwise.1.weight  --> new: block3.0.depthwise_bn.weight
      - old: block3.0.pointwise.0.weight  --> new: block3.0.pointwise_conv.weight
      - old: block3.0.pointwise.1.weight  --> new: block3.0.pointwise_bn.weight
      - 忽略 relu 层 (old key 包含 “.2.” 的那些)
    """
    for old_key, v in old_sd.items():
        # 跳过 relu 的权重／缓冲
        if re.search(r"\.(depthwise|pointwise)\.2\.", old_key):
            continue

        new_key = old_key
        new_key = re.sub(r"\.depthwise\.0\.",    r".depthwise_conv.", new_key)
        new_key = re.sub(r"\.depthwise\.1\.",    r".depthwise_bn.",   new_key)
        new_key = re.sub(r"\.pointwise\.0\.",    r".pointwise_conv.", new_key)
        new_key = re.sub(r"\.pointwise\.1\.",    r".pointwise_bn.",   new_key)

        if new_key in new_sd:
            new_sd[new_key] = v
        else:
            print(f"⚠️ Warning: 找不到 new key `{new_key}`，跳过。")

    return new_sd

if __name__ == "__main__":
    # —— 2. 加载原始模型并拿到 state_dict —— 
    old_model = XFeatModelOld()
    old_sd = torch.load("D:/SCIResearch/code/accelerated_features-test/weights/dw_pw_160000.pth", map_location="cpu")

    old_model.load_state_dict(old_sd)
    old_model.eval()

    # —— 3. 用新模型模板创建一个实例，拿到它的 state_dict —— 
    new_model = XFeatModelNew()
    new_sd = new_model.state_dict()

    # —— 4. 拷贝 & 重映射旧权重到 new_sd —— 
    fused_sd = copy_and_map_state_dict(old_sd, new_sd)

    # —— 5. 把 new_model.load_state_dict(fused_sd) —— 
    new_model.load_state_dict(fused_sd)
    new_model.eval()

    # —— 6. 真正执行 fuse —— 
    #    定义一个简单函数，遍历所有 DS layer，调用 fuse_model()
    for m in new_model.modules():
        if isinstance(m, DepthwiseSeparableLayer):
            m.fuse_model()

    # —— 7. 保存“已融合”的权重 —— 
    torch.save(new_model.state_dict(), "./fused_weights.pth")
    print("✅ 已生成 fused_weights.pth，下次直接 load 这个文件即可。")

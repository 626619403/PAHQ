#!/usr/bin/env python
# universal_edge_connector.py
#
# 本文件用于归纳之前四个文件中对不同任务模型（docstring、greaterthan、ioi、induction）
# 节点边生成的方式，并推广到结构和规模类似的其他模型上。程序会根据所传入的任务类型，
# 加载对应的模型与连接生成函数，最终输出一个连接边的集合（字典）。
#
# 使用示例：
#    python universal_edge_connector.py --task docstring --device cuda
#    python universal_edge_connector.py --task greaterthan --device cpu

import argparse
import torch

# 如果需要扩展到其他模型，可以在这里提前定义每个任务类型默认模型加载函数
def load_docstring_model(device):
    try:
        # 来自第一个文件：docstring 模型加载与设置
        from transformer_lens import HookedTransformer
    except ImportError:
        raise ImportError("请确认 transformer_lens 模块可用")
    tl_model = HookedTransformer.from_pretrained("attn-only-4l")
    tl_model.set_use_attn_result(True)
    tl_model.set_use_split_qkv_input(True)
    tl_model.to(device)
    return tl_model

def load_greaterthan_model(device):
    try:
        # 来自第二个文件中 greaterthan 使用的是 get_gpt2_small
        from acdc.ioi.utils import get_gpt2_small
    except ImportError:
        raise ImportError("请确认 acdc.ioi.utils 模块可用")
    return get_gpt2_small(device=device)

def load_ioi_model(device):
    try:
        # 来自第三个文件，IOI 任务也用 get_gpt2_small
        from acdc.ioi.utils import get_gpt2_small
    except ImportError:
        raise ImportError("请确认 acdc.ioi.utils 模块可用")
    return get_gpt2_small(device=device)

def load_induction_model(device):
    try:
        # 可复用 redwood_attn_2l 作为 induction 任务的模型，也可以替换成其他相似模型
        from transformer_lens import HookedTransformer
    except ImportError:
        raise ImportError("请确认 transformer_lens 模块可用")
    tl_model = HookedTransformer.from_pretrained("redwood_attn_2l", center_writing_weights=False, center_unembed=False, fold_ln=False, device=device)
    tl_model.set_use_attn_result(True)
    tl_model.set_use_split_qkv_input(True)
    return tl_model

# 定义每个任务的边生成函数
def generate_docstring_edges(model, device):
    """
    根据 docstring 任务的连接方式生成边集。
    使用 get_docstring_subgraph_true_edges 来构建手工定义的边。
    """
    try:
        from acdc.docstring.utils import get_docstring_subgraph_true_edges
    except ImportError:
        raise ImportError("无法导入 acdc.docstring.utils 模块，请检查路径设置")
    edges = get_docstring_subgraph_true_edges()
    return edges

def generate_greaterthan_edges(model, device):
    """
    根据 greaterthan 任务的连接方式生成边集。
    此处调用 get_greaterthan_true_edges，需要传入模型。
    """
    try:
        from acdc.ioi.utils import get_greatERthan_true_edges  # 注意，有可能函数名为 get_greaterthan_true_edges
    except ImportError:
        # 如果上述模块未定义，我们尝试关键词匹配名称
        try:
            from acdc.ioi.utils import get_greaterthan_true_edges as func
            get_edges = func
        except ImportError:
            raise ImportError("无法导入 greaterthan 相关边生成函数，请检查路径")
    else:
        get_edges = get_greatERthan_true_edges

    # 这里model为加载好的模型
    edges = get_edges(model)
    return edges

def generate_ioi_edges(model, device):
    """
    根据 IOI 任务的连接方式生成边集。
    调用 get_ioi_true_edges 函数，传入模型。
    """
    try:
        from acdc.ioi.utils import get_ioi_true_edges
    except ImportError:
        raise ImportError("无法导入 IOI 任务的边生成函数，请检查模块路径")
    edges = get_ioi_true_edges(model)
    return edges

def generate_induction_edges(model, device):
    """
    对于 induction 任务，由于原始代码中没有直接给出生成边的函数，
    我们这里提供一种推广策略：
      根据模型中各层、各位置构造“虚拟”连接边。
    例如：假设 induction 模型与 redwood_attn_2l 相似，具有 L 层，每层 MLP、Attention 等节点，
          我们可以将每个层的隐含节点与上层、下层的节点做全连接（或者采用其他启发式策略）
    注意：你需要根据实际需求对此函数进行修改。
    """
    # 以下仅为示例：假设模型有 12 层，每层各有 1 个 MLP 节点和 12 个 attention head 节点
    edge_dict = {}
    num_layers = getattr(model.cfg, "n_layers", 12)
    num_heads = getattr(model.cfg, "n_heads", 12)
    # 构造 MLP 之间连接：例如连接第 i 层 MLP -> 第 i+1 层 MLP
    for i in range(num_layers - 1):
        key = (f"blocks.{i}.hook_mlp_out", (i,), f"blocks.{i+1}.hook_mlp_in", (i+1,))
        edge_dict[key] = True

    # 构造 Attention 内连接：例如每层所有 head 之间全连（可根据需要调整）
    for layer in range(num_layers):
        for h1 in range(num_heads):
            for h2 in range(num_heads):
                if h1 != h2:
                    key = (f"blocks.{layer}.attn.hook_result", (layer, h1), f"blocks.{layer}.attn.hook_{'q'}_input", (layer, h2))
                    edge_dict[key] = True

    # 构造跨层 Attention 连接（例如当前层的 attention 输出与下一层的 attention 输入之间建立连接）
    for layer in range(num_layers - 1):
        for head in range(num_heads):
            key = (f"blocks.{layer}.attn.hook_result", (layer, head), f"blocks.{layer+1}.attn.hook_q_input", (layer+1, head))
            edge_dict[key] = True

    # 返回该“虚拟”边集合
    return edge_dict

# 定义一个统一的接口，根据任务类型生成边集
def generate_edge_set(task, device="cuda"):
    task = task.lower()
    if task == "docstring":
        model = load_docstring_model(device)
        edges = generate_docstring_edges(model, device)
    elif task == "greaterthan":
        model = load_greaterthan_model(device)
        edges = generate_greaterthan_edges(model, device)
    elif task == "ioi":
        model = load_ioi_model(device)
        edges = generate_ioi_edges(model, device)
    elif task == "induction":
        model = load_induction_model(device)
        edges = generate_induction_edges(model, device)
    else:
        raise ValueError(f"未知任务类型：{task}。请使用 docstring, greaterthan, ioi 或 induction")
    return edges

def main():
    parser = argparse.ArgumentParser(
        description="根据模型任务类型生成节点之间的边集(自动推广到类似规模的模型)"
    )
    parser.add_argument("--task", type=str, required=True,
                        help="任务类型，支持: docstring, greaterthan, ioi, induction")
    parser.add_argument("--device", type=str, default="cuda", help="设备选择，默认为 cuda")
    args = parser.parse_args()

    print("正在加载任务：", args.task)
    edges = generate_edge_set(args.task, args.device)
    print("连接后的边集共计：", len(edges), "条")
    # 输出边集信息（仅打印前若干条，防止信息过多）
    count = 0
    for key, present in edges.items():
        print(key, "=>", present)
        count += 1
        if count >= 10:
            print("…… 其他边略")
            break

    # 返回（或者写入文件）最终的边集
    return edges

if __name__ == "__main__":
    main()


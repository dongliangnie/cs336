from __future__ import annotations

import os
from collections.abc import Iterable
from typing import IO, Any, BinaryIO

import numpy.typing as npt
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor


def run_linear(
    d_in: int,
    d_out: int,
    weights: Float[Tensor, " d_out d_in"],
    in_features: Float[Tensor, " ... d_in"],
) -> Float[Tensor, " ... d_out"]:
    """
    给定线性层的权重，计算批次输入的线性变换。

    参数:
        d_in (int): 输入维度的尺寸 (in_dim)
        d_out (int): 输出维度的尺寸 (out_dim)
        weights (Float[Tensor, "d_out d_in"]): 使用的线性层权重矩阵
        in_features (Float[Tensor, "... d_in"]): 需要应用该线性变换的输入张量

    返回:
        Float[Tensor, "... d_out"]: 线性模块变换后的输出张量。
    """

    raise NotImplementedError


def run_embedding(
    vocab_size: int,
    d_model: int,
    weights: Float[Tensor, " vocab_size d_model"],
    token_ids: Int[Tensor, " ..."],
) -> Float[Tensor, " ... d_model"]:
    """
    给定嵌入层的权重，获取一批 Token ID 对应的嵌入向量。

    参数:
        vocab_size (int): 词表中的词条/嵌入向量总数
        d_model (int): 嵌入维度的尺寸
        weights (Float[Tensor, "vocab_size d_model"]): 用于查表的嵌入矩阵
        token_ids (Int[Tensor, "..."]): 需要从嵌入层中提取的 Token ID 集合

    返回:
        Float[Tensor, "... d_model"]: 嵌入层返回的批次嵌入向量。
    """

    raise NotImplementedError


def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, " d_ff d_model"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    w3_weight: Float[Tensor, " d_ff d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """给定 SwiGLU 网络的权重，返回使用这些权重计算得到的输出。

    参数:
        d_model (int): 输入和输出特征的隐藏层维度。
        d_ff (int): SwiGLU 内部前向传播时的升维/中间层维度。
        w1_weight (Float[Tensor, "d_ff d_model"]): 存储的 W1 权重
        w2_weight (Float[Tensor, "d_model d_ff"]): 存储的 W2 权重
        w3_weight (Float[Tensor, "d_ff d_model"]): 存储的 W3 权重
        in_features (Float[Tensor, "... d_model"]): 输入到该前馈层的嵌入表示。

    返回:
        Float[Tensor, "... d_model"]: 与输入嵌入张量形状完全相同的输出嵌入表示。
    """
    # 示例:
    # 如果你的 state dict 键值对匹配，可以直接使用 `load_state_dict()`
    # swiglu.load_state_dict(weights)
    # 你也可以手动对其进行权重赋值
    # swiglu.w1.weight.data = w1_weight
    # swiglu.w2.weight.data = w2_weight
    # swiglu.w3.weight.data = w3_weight
    raise NotImplementedError


def run_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... keys d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """
    给定查询 (Q)、键 (K) 和值 (V) 张量，返回缩放点积注意力（SDPA）的输出。

    参数:
        Q (Float[Tensor, " ... queries d_k"]): 查询 (Query) 张量
        K (Float[Tensor, " ... keys d_k"]): 键 (Key) 张量
        V (Float[Tensor, " ... keys d_v"]): 值 (Value) 张量
        mask (Bool[Tensor, " ... queries keys"] | None): 掩码 (Mask) 张量
    返回:
        Float[Tensor, " ... queries d_v"]: 缩放点积注意力机制的输出
    """
    raise NotImplementedError


def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, " d_model d_model"],
    k_proj_weight: Float[Tensor, " d_model d_model"],
    v_proj_weight: Float[Tensor, " d_model d_model"],
    o_proj_weight: Float[Tensor, " d_model d_model"],
    in_features: Float[Tensor, " ... sequence_length d_model"],
) -> Float[Tensor, " ... sequence_length d_model"]:
    """
    根据多头注意力朴素单序列（未分批）实现的 Q、K、V 投影权重，返回优化后的分批实现输出。
    该实现应在单个矩阵乘法中，同时处理所有头（Heads）的键、查询和值投影。
    此版本的函数不包含 RoPE（旋转位置编码）。
    参考 Vaswani 等人 2017 年论文的第 3.2.2 节。

    参数:
        d_model (int): 输入和输出特征的隐藏层维度。
        num_heads (int): 多头注意力机制中使用的头数。
        max_seq_len (int): 如果你的实现中包含预缓存（Pre-cache）机制，此值代表预缓存的最大序列长度。
        q_proj_weight (Float[Tensor, "d_model d_model"]): Q 投影的权重
        k_proj_weight (Float[Tensor, "d_model d_model"]): K 投影的权重
        v_proj_weight (Float[Tensor, "d_model d_model"]): V 投影的权重
        o_proj_weight (Float[Tensor, "d_model d_model"]): 输出（Output）投影的权重
        in_features (Float[Tensor, "... sequence_length d_model"]): 运行该多头注意力的输入特征张量。

    返回:
        Float[Tensor, " ... sequence_length d_model"]: 包含优化分批多头自注意力输出的张量。
    """
    raise NotImplementedError


def run_multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Float[Tensor, " d_model d_model"],
    k_proj_weight: Float[Tensor, " d_model d_model"],
    v_proj_weight: Float[Tensor, " d_model d_model"],
    o_proj_weight: Float[Tensor, " d_model d_model"],
    in_features: Float[Tensor, " ... sequence_length d_model"],
    token_positions: Int[Tensor, " ... sequence_length"] | None = None,
) -> Float[Tensor, " ... sequence_length d_model"]:
    """
    根据多头注意力朴素单序列（未分批）实现的 Q、K、V 投影权重，返回优化后的分批实现输出。
    该实现应在单个矩阵乘法中，同时处理所有头（Heads）的键、查询和值投影。
    此版本的多头注意力（MHA）应包含 RoPE（旋转位置编码）。
    在这种情况下，RoPE 作用的维度必须是每个头的嵌入维度 (d_model // num_heads)。
    参考 Vaswani 等人 2017 年论文的第 3.2.2 节。

    参数:
        d_model (int): 输入和输出特征的隐藏层维度。
        num_heads (int): 多头注意力机制中使用的头数。
        max_seq_len (int): 如果你的实现中包含预缓存（Pre-cache）机制，此值代表预缓存的最大序列长度。
        theta (float): RoPE 的底数参数 Theta。
        q_proj_weight (Float[Tensor, "d_model d_model"]): Q 投影的权重
        k_proj_weight (Float[Tensor, "d_model d_model"]): K 投影的权重
        v_proj_weight (Float[Tensor, "d_model d_model"]): V 投影的权重
        o_proj_weight (Float[Tensor, "d_model d_model"]): 输出（Output）投影的权重
        in_features (Float[Tensor, "... sequence_length d_model"]): 运行该多头注意力的输入特征张量。
        token_positions (Int[Tensor, " ... sequence_length"] | None): 可选张量，包含每个 Token 的绝对位置索引

    返回:
        Float[Tensor, " ... sequence_length d_model"]: 包含优化分批并应用 RoPE 后的多头自注意力输出张量。
    """
    raise NotImplementedError


def run_rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Float[Tensor, " ... sequence_length d_k"],
    token_positions: Int[Tensor, " ... sequence_length"],
) -> Float[Tensor, " ... sequence_length d_k"]:
    """
    为给定的输入张量（Query 或 Key）计算并应用旋转位置编码（RoPE）。

    参数:
        d_k (int): 查询或键张量的头嵌入维度尺寸。
        theta (float): RoPE 的底数参数 Theta。
        max_seq_len (int): 如果你的实现中包含预缓存机制，此值代表预缓存的最大序列长度。
        in_query_or_key (Float[Tensor, "... sequence_length d_k"]): 需要注入 RoPE 的输入张量。
        token_positions (Int[Tensor, "... sequence_length"]): 形状为 (batch_size, sequence_length) 的 Token 位置张量
    返回:
        Float[Tensor, " ... sequence_length d_k"]: 完成旋转位置编码（RoPE）编码后的输出张量。
    """
    raise NotImplementedError


def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, " batch sequence_length d_model"],
) -> Float[Tensor, " batch sequence_length d_model"]:
    """
    给定 Pre-Norm Transformer 块的权重和输入特征，返回在输入特征上运行 Transformer 块后的输出。

    该函数应当使用 RoPE（旋转位置编码）。
    取决于你的具体实现，你可能只需要将相关的参数传递给 TransformerBlock 的构造函数，
    或者你需要手动初始化你自己的 RoPE 类并将其传入。

    参数:
        d_model (int): Transformer 块输入的特征维度。
        num_heads (int): 多头注意力中使用的头数。 `d_model` 必须能被 `num_heads` 整除。
        d_ff (int): 前馈网络（FFN）内部中间层的维度。
        max_seq_len (int): 如果你的实现中包含预缓存机制，此值代表预缓存的最大序列长度。
        theta (float): RoPE 的底数参数 Theta。
        weights (dict[str, Tensor]):
            基准参考实现的权重字典（State Dict）。
            该字典的键包含:
            - `attn.q_proj.weight`
                所有 `num_heads` 个注意力头的查询（Query）投影矩阵。
                形状为 (d_model, d_model)。
                行顺序由形状为 (num_heads, d_k) 的子矩阵顺次拼接而成，
                即 `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`。
            - `attn.k_proj.weight`
                所有 `num_heads` 个注意力头的键（Key）投影矩阵。
                形状为 (d_model, d_model)。
                行顺序由形状为 (num_heads, d_k) 的子矩阵顺次拼接而成，
                即 `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`。
            - `attn.v_proj.weight`
                所有 `num_heads` 个注意力头的值（Value）投影矩阵。
                形状为 (d_model, d_model)。
                行顺序由形状为 (num_heads, d_v) 的子矩阵顺次拼接而成，
                即 `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`。
            - `attn.output_proj.weight`
                多头自注意力输出投影（O 投影）的权重矩阵。
                形状为 (d_model, d_model)。
            - `ln1.weight`
                Transformer 块中应用的第一个 RMSNorm 的仿射变换权重。
                形状为 (d_model,)。
            - `ffn.w1.weight`
                FFN（前馈网络）中第一个线性变换的权重。
                形状为 (d_ff, d_model)。
            - `ffn.w2.weight`
                FFN 中第二个线性变换的权重。
                形状为 (d_model, d_ff)。
            - `ffn.w3.weight`
                FFN 中第三个线性变换的权重。
                形状为 (d_ff, d_model)。
            - `ln2.weight`
                Transformer 块中应用的第二个 RMSNorm 的仿射变换权重。
                形状为 (d_model,)。
        in_features (Float[Tensor, "batch sequence_length d_model"]):
            用于运行当前模块的输入特征张量。

    返回:
        Float[Tensor, "batch sequence_length d_model"] 在输入特征上运行并包含 RoPE 处理后的 Transformer 块输出张量。
    """
    raise NotImplementedError


def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Int[Tensor, " batch_size sequence_length"],
) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
    """给定 Transformer 语言模型的权重和输入 Token 索引，返回运行前向传播后的输出。

    该函数应当使用 RoPE（旋转位置编码）。

    参数:
        vocab_size (int): 待预测的输出词表中唯一 Token 的总数。
        context_length (int): 模型单次能处理的最大 Token 上下文序列长度。
        d_model (int): 模型嵌入层以及各个子层输出的隐藏特征维度。
        num_layers (int): 使用的 Transformer 层的总层数。
        num_heads (int): 多头注意力中使用的头数。 `d_model` 必须能被 `num_heads` 整除。
        d_ff (int): 前馈网络内部中间层的特征维度（第 3.3 节）。
        rope_theta (float): RoPE 的 Theta 参数。
        weights (dict[str, Tensor]):
            基准参考实现的权重字典（State Dict）。其中的 {num_layers} 代表
            介于 `0` 到 `num_layers - 1` 之间的整数（层索引）。
            该字典的键包含:
            - `token_embeddings.weight`
                Token 嵌入矩阵。形状为 (vocab_size, d_model)。
            - `layers.{num_layers}.attn.q_proj.weight`
                第 num_layers 层中所有 `num_heads` 个注意力头的查询投影矩阵。
                形状为 (num_heads * (d_model / num_heads), d_model)。
                行顺序由矩阵 (num_heads, d_k) 顺次拼接而成，
                即 `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`。
            - `layers.{num_layers}.attn.k_proj.weight`
                第 num_layers 层中所有 `num_heads` 个注意力头的键投影矩阵。
                形状为 (num_heads * (d_model / num_heads), d_model)。
                行顺序由矩阵 (num_heads, d_k) 顺次拼接而成，
                即 `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`。
            - `layers.{num_layers}.attn.v_proj.weight`
                第 num_layers 层中所有 `num_heads` 个注意力头的值投影矩阵。
                形状为 (num_heads * (d_model / num_heads), d_model)。
                行顺序由矩阵 (num_heads, d_v) 顺次拼接而成，
                即 `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`。
            - `layers.{num_layers}.attn.output_proj.weight`
                第 num_layers 层中多头自注意力输出投影的权重。
                形状为 ((d_model / num_heads) * num_heads, d_model)。
            - `layers.{num_layers}.ln1.weight`
                在第一层 Transformer 块中应用的第一个 RMSNorm 的仿射变换权重。
                形状为 (d_model,)。
            - `layers.{num_layers}.ffn.w1.weight`
                第 num_layers 层 FFN 中第一个线性变换的权重。
                形状为 (d_ff, d_model)。
            - `layers.{num_layers}.ffn.w2.weight`
                第 num_layers 层 FFN 中第二个线性变换的权重。
                形状为 (d_model, d_ff)。
            - `layers.{num_layers}.ffn.w3.weight`
                第 num_layers 层 FFN 中第三个线性变换的权重。
                形状为 (d_ff, d_model)。
            - `layers.{num_layers}.ln2.weight`
                在第一层 Transformer 块中应用的第二个 RMSNorm 的仿射变换权重。
                形状为 (d_model,)。
            - `ln_final.weight`
                应用在最后一层 Transformer 块输出之后的最终 RMSNorm 的仿射变换权重。
                形状为 (d_model, )。
            - `lm_head.weight`
                语言模型输出词表嵌入层（预测 Logits）的权重。
                形状为 (vocab_size, d_model)。
        in_indices (Int[Tensor, "batch_size sequence_length"]): 用于运行语言模型的输入 Token 索引张量。形状为 (batch_size, sequence_length)，其中
            `sequence_length` 最大不超过 `context_length`。

    返回:
        Float[Tensor, "batch_size sequence_length vocab_size"]: 包含每个 Token 位置对应的未归一化下一词概率分布（Logits）的张量。
    """
    raise NotImplementedError


def run_rmsnorm(
    d_model: int,
    eps: float,
    weights: Float[Tensor, " d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """给定 RMSNorm 仿射变换的权重，返回在输入特征上运行 RMSNorm 后的输出。

    参数:
        d_model (int): RMSNorm 输入的特征维度。
        eps: (float): 添加到分母中以保证数值稳定性的微小值。
        weights (Float[Tensor, "d_model"]): RMSNorm 的缩放权重（Gamma）。
        in_features (Float[Tensor, "... d_model"]): 需要进行 RMSNorm 的输入特征张量，可以拥有任意数量的前导维度（Batch 维度）。

    返回:
        Float[Tensor,"... d_model"]: 与 `in_features` 形状完全相同、完成 RMSNorm 归一化后的输出张量。
    """
    raise NotImplementedError


def run_silu(in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    """给定一个输入张量，返回对每个元素应用 SiLU（Sigmoid 线性单元）激活函数后的输出。

    参数:
        in_features(Float[Tensor, "..."]): 需要运行 SiLU 的输入特征张量。形状不限。

    返回:
        Float[Tensor,"..."]: 结构和形状与 `in_features` 完全相同的 SiLU 激活输出张量。
    """
    raise NotImplementedError


def run_get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    给定数据集（一个由整数 Token ID 组成的一维 NumPy 数组）以及所需的批次大小和上下文长度，
    从中随机采样用于语言模型训练的前向序列输入（Inputs）以及它们对应的标签（Labels）。

    参数:
        dataset (np.array): 数据集中由整型 Token ID 构成的一维 NumPy 数组。
        batch_size (int): 期望采样的批次大小 (Batch Size)。
        context_length (int): 每个采样样本所期望的上下文长度 (Sequence Length)。
        device (str): PyTorch 设备字符串（例如 'cpu' 或 'cuda:0'），指定采样后的输入序列和标签应放置在哪个硬件设备上。

    返回:
        形状为 (batch_size, context_length) 的 torch.LongTensor 元组。元组的第一个元素
        是采样得到的输入序列，第二个元素是对应的语言模型训练真实标签（通常为输入序列向后平移一位）。
    """
    raise NotImplementedError


def run_softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    """
    给定一个输入张量，返回对输入指定维度（dim）进行 Softmax 归一化后的输出。

    参数:
        in_features (Float[Tensor, "..."]): 需要应用 Softmax 的输入特征张量。形状不限。
        dim (int): `in_features` 中执行 Softmax 归一化的目标维度。

    返回:
        Float[Tensor, "..."]: 与 `in_features` 形状完全相同、指定维度被 Softmax 归一化后的输出张量。
    """
    raise NotImplementedError


def run_cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    """给定输入（Logits）和目标标签（Targets），计算所有样本的平均交叉熵损失（Cross-Entropy Loss）。

    参数:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] 代表第 i 个样本在第 j 个类别上的未归一化预测对数几率 (Logit)。
        targets (Int[Tensor, "batch_size"]): 形状为 (batch_size,) 的张量，包含正确类别的索引值。
            每个索引值必须介于 0 到 `num_classes - 1` 之间。

    返回:
        Float[Tensor, ""]: 所有样本的平均交叉熵损失标量张量。
    """
    raise NotImplementedError


def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """给定一组参数，裁剪它们的组合梯度，使其 L2 范数最多不超过 max_l2_norm。

    参数:
        parameters (Iterable[torch.nn.Parameter]): 可训练参数的集合。
        max_l2_norm (float): 梯度的最大 L2 范数阈值（正数）。

    参数的梯度（parameter.grad）应当被原地（in-place）修改。
    """
    raise NotImplementedError


def get_adamw_cls() -> Any:
    """
    返回一个实现了 AdamW 优化算法的 torch.optim.Optimizer 类。
    """
    raise NotImplementedError


def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    """
    给定余弦学习率衰减策略（带线性预热）的各项参数以及当前的迭代步数，
    返回在该特定迭代步数下由策略调度决定的学习率。

    参数:
        it (int): 获取当前学习率对应的具体迭代步数（Iteration number）。
        max_learning_rate (float): alpha_max，余弦学习率调度中的最大学习率（预热结束后的峰值）。
        min_learning_rate (float): alpha_min，余弦学习率调度中的最小/最终落地学习率。
        warmup_iters (int): T_w，用于线性预热（Warm-up）学习率的迭代步数。
        cosine_cycle_iters (int): T_c，余弦退火（Annealing）阶段持续的迭代步数。

    返回:
        当前迭代步数在指定调度策略下的学习率数值。
    """
    raise NotImplementedError


def run_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    给定模型、优化器和当前迭代步数，将它们的状态序列化并保存到磁盘中（存盘点）。

    参数:
        model (torch.nn.Module): 需要序列化其状态字典（State Dict）的模型。
        optimizer (torch.optim.Optimizer): 需要序列化其状态字典的优化器。
        iteration (int): 需要一并序列化的整数，代表当前已经完成的训练迭代总步数。
        out (str | os.PathLike | BinaryIO | IO[bytes]): 保存模型、优化器和迭代步数的磁盘路径或类文件对象（File-like object）。
    """
    raise NotImplementedError


def run_load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """
    给定一个序列化好的检查点（磁盘路径或类文件对象），将序列化的状态恢复到指定的模型和优化器中。
    返回检查点文件中先前序列化保存的训练迭代步数。

    参数:
        src (str | os.PathLike | BinaryIO | IO[bytes]): 序列化检查点文件的路径或类文件对象。
        model (torch.nn.Module): 接收并恢复状态的模型。
        optimizer (torch.optim.Optimizer): 接收并恢复状态的优化器。
    返回:
        int: 先前序列化时保存的训练迭代总步数。
    """
    raise NotImplementedError


def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Any:
    """给定词表、合并规则列表和特殊令牌列表，返回一个基于这些配置构建的 BPE（Byte Pair Encoding）分词器。

    参数:
        vocab (dict[int, bytes]): 分词器词表，一个从整数（词表中的 Token ID）到原始字节（Token 字节串）的映射字典。
        merges (list[tuple[bytes, bytes]]): BPE 合并规则。列表中的每一项都是一个原始字节元组 (<token1>, <token2>)，
            代表 <token1> 与 <token2> 被合并。
            合并规则按照它们被创建/学习到的先后顺序进行排列。
        special_tokens (list[str] | None): 分词器专属的字符串特殊令牌列表。这些特殊字符串在分词时绝不会被拆分为多个 Token，
            而会始终作为一个整体 Token 被保留。

    返回:
        一个配置了指定词表、合并规则和特殊令牌的 BPE 分词器实例。
    """
    from cs336_basics.tokenizer_encoder import Tokenizer
    return Tokenizer(
        vocab=vocab,
        merges=merges,
        special_tokens=special_tokens,
    )


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """给定输入语料库的路径，训练一个 BPE 分词器，并输出训练得到的词表和合并规则。

    参数:
        input_path (str | os.PathLike): 用于训练 BPE 分词器的输入文本数据路径。
        vocab_size (int): 分词器词表所期望的总大小（包含特殊令牌在内）。
        special_tokens (list[str]): 需要显式添加进分词器词表的特殊令牌字符串列表。
            这些特殊字符串在分词时绝不会被拆分为多个 Token，而会始终作为一个整体 Token
            被保留。如果这些特殊令牌出现在 `input_path` 的语料中，它们会被当做普通字符串一并处理。

    返回:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                训练完成的分词器词表，一个从整数（词表中的 Token ID）到原始字节（Token 字节串）的映射字典。
            merges:
                BPE 合并规则。列表中的每一项都是一个原始字节元组 (<token1>, <token2>)，
                代表 <token1> 与 <token2> 被合并。
                合并规则按照它们被创建/学习到的先后顺序进行排列。
    """
    from cs336_basics.bpe_tokenzier import run_train_bpe as run_train_bpe
    return run_train_bpe(
        input_path=input_path,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        **kwargs,
    )
    raise NotImplementedError

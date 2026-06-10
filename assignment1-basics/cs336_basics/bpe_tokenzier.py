import os
import collections
import json
import pickle
import regex

from collections import defaultdict
from typing import (
    List,
    Tuple,
    Dict,
    Set,
    Union,
)

def merge_token_sequence(token_seq: Tuple[bytes, ...], best_pair: Tuple[bytes, bytes], new_token: bytes) -> Tuple[bytes, ...]:
    """
    在一个 Token 字节序列中，将所有连续出现的最佳词对 (best_pair) 融合成一个新的单一 Token。

    参数:
        token_seq: 原始的由基础字节片段组成的元组，例如 (b'h', b'e', b'l', b'l', b'o')
        best_pair: 当前轮次被选中的高频连续相邻词对，例如 (b'l', b'l')
        new_token: 融合后的新字节串，例如 b'll'

    返回:
        融合后的新元组，例如 (b'h', b'e', b'll', b'o')
    """
    new_seq = []
    i = 0
    while i < len(token_seq):
        # 双指针/滑动窗口检查：若当前位置及下一位置构成的词对正好匹配 best_pair，则执行合并
        if i < len(token_seq) - 1 and (token_seq[i], token_seq[i+1]) == best_pair:
            new_seq.append(new_token)
            i += 2  # 跳过已被合并的下一个元素，防止重复消费
        else:
            # 否则原封不动保留当前 Token
            new_seq.append(token_seq[i])
            i += 1
    return tuple(new_seq)

def run_train_bpe(
    input_path: Union[str, os.PathLike],
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    给定训练语料库的路径，运行 BPE (Byte Pair Encoding) 算法进行分词器训练，
    并输出最终构建的词汇表 (vocab) 和合并规则记录 (merges)。

    参数:
        input_path: BPE 训练文本文件的路径。
        vocab_size: 期望的目标词汇表总大小（包含 256 个基础字节、特殊符号以及后续合并产生的新 Token）。
        special_tokens: 显式注入的特殊符号列表（如 "<|endoftext|>"），在语料切分中作为绝对隔离边界，不可被拆分。

    返回:
        vocab: 字典，键为 Token ID (int)，值为对应的字节串 (bytes)。
        merges: 列表，每个元素为被合并的词对元组 (bytes, bytes)，顺序代表了合并规则的先后优先级。
    """

    # ==========================================
    # 第 0 步: 参数鲁棒性校验
    # ==========================================
    if not isinstance(vocab_size, int) or vocab_size <= 0:
        raise ValueError("vocab_size 必须是一个正整数。")

    # ==========================================
    # 第 1 步: 初始化基础词汇表
    # ==========================================
    # 基础词汇表必须完整映射所有 256 个单字节情况（0x00 ~ 0xFF），确保分词器永远不会遇到 OOV (Out-of-Vocabulary) 错误。
    # 这里的 bytes([i]) 将 0-255 的整数直接转换为长度为 1 的单字节序列（例如 65 -> b'A'）。
    vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    current_next_id: int = 256  # 融合生成的新 Token ID 将从 256 开始递增

    # 用集合 (Set) 来维护当前词汇表中已有的所有字节流值，以便在 O(1) 复杂度下进行查重
    existing_byte_values: Set[bytes] = set(vocab.values())

    # 将外部传入的特殊符号（Special Tokens）静态注入到词汇表中
    for st_str in special_tokens:
        if len(vocab) >= vocab_size:  # 若词汇表容量已被基础字节或前面的特殊符填满，则终止注入
            break
        st_bytes = st_str.encode("utf-8")  # 将特殊符号文本序列化为 UTF-8 字节流
        
        # 查重防护：避免重复插入与基础字节冲突的特殊符（例如若传入特殊符 "a"，其编码为 b'a'，已存在于前 256 个基础字节中）
        if st_bytes not in existing_byte_values:
            vocab[current_next_id] = st_bytes
            existing_byte_values.add(st_bytes)
            current_next_id += 1  # 递增 ID 计数器

    # ==========================================
    # 第 2 步: 加载语料数据
    # ==========================================
    try:
        # 使用 errors="ignore" 防止由于非 UTF-8 坏字符导致读取中断，增强对不干净语料的包容性
        with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
    except FileNotFoundError:
        text = ""  # 路径失效时鲁棒降级为空字符串，避免直接崩溃

    # ==========================================
    # 第 3 步: 预分词处理 (Pre-tokenization)
    # ==========================================
    # 3.1 隔离特殊符号：利用 regex.escape 将特殊符号转义，并用 '|' 拼接成正则选择分支。
    # 将文本按照特殊符号切分成大块 (Chunks)，确保特殊符号内部不会被后续的 FFN/标点正则切碎。
    chunks = regex.split('|'.join(map(regex.escape, special_tokens)), text)

    # 3.2 引入 GPT-2/GPT-4 标准的预分词正则表达式 (PAT)。该正则的作用是在字节融合前，
    # 将文本切分为合理的“字词”片段，防止空格与文本、数字与字母、不同标点之间发生跨类别合并。
    # - '(?:[sdmt]|ll|ve|re) : 匹配常见的英文缩写（如 's, 'd, 'll 等）
    # -  ?\p{L}+            : 匹配可选带前导空格的连续字母序列（支持多国语言字符）
    # -  ?\p{N}+            : 匹配可选带前导空格的连续数字序列
    # -  ?[^\s\p{L}\p{N}]+  : 匹配可选带前导空格的连续标点/特殊符号串（不含空格、字母、数字）
    # - \s+(?!\S)           : 匹配文末或独立的连续空白字符（不含非空白字符）
    # - \s+                 : 匹配其他连续空白
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    
    # 核心频次字典：统计“单词内部字节元组序列”在全局语料中的出现频次
    # 键必须是不可变的 tuple，例如 (b'h', b'e', b'l', b'l', b'o') -> 频次
    token_frequency_table = defaultdict(int)

    for chunk in chunks:
        if not chunk:
            continue
        for word in regex.findall(PAT, chunk):
            word_bytes = word.encode("utf-8")  # 将当前匹配到的单词块转化为原始字节流
            # 将每个单字节抽取出来，包装成单字节序列的列表，例如 b'hi' -> [b'h', b'i']
            bytes_list = [bytes([x]) for x in word_bytes]
            # 转换为元组并累计频次
            token_frequency_table[tuple(bytes_list)] += 1

    # 3.3 全局词对（相邻两个 Token）频次统计字典
    # 键为 (Token_A, Token_B) 的双元组，值为全局累计频次
    pair_counts = defaultdict(int)
    for token_seq, freq in token_frequency_table.items():
        for i in range(len(token_seq) - 1):
            pair_counts[token_seq[i], token_seq[i+1]] += freq

    # ==========================================
    # 第 4 步: BPE 核心循环训练（迭代合并高频对）
    # ==========================================
    merges: List[Tuple[bytes, bytes]] = []  # 顺序记录每一步的合并规则

    while len(vocab) < vocab_size:
        if not pair_counts:  # 如果所有单词都被聚合成单一 Token 或没有相邻词对可供合并，提前结束循环
            break

        # 4.1 寻找当前语料中最常出现的相邻词对
        max_count = max(pair_counts.values())
        # 获取所有频次等于 max_count 的候选词对列表
        candidates = [k for k, v in pair_counts.items() if v == max_count]
        
        # 4.2 决胜策略 (Tie-breaking): 
        # 如果存在多个词对频次并列第一，选择字节序（Byte Order）最大的那个对（max(candidates)）。
        # Python 在对由 bytes 组成的元组求 max 时，会自左向右逐个比较字节的数值大小，确保了策略的确定性与可复现性。
        best_pair = max(candidates)
        merges.append(best_pair)  # 记录当前合并规则

        # 4.3 构造合并后的新 Token 字节流（直接将相邻两个 Token 的字节串进行拼接）
        new_token_bytes = best_pair[0] + best_pair[1]
        vocab[current_next_id] = new_token_bytes  # 写入词汇表
        current_next_id += 1

        # 4.4 高效增量更新：找出所有包含当前 best_pair 的词条序列
        # 我们不能每一轮都重新去扫描全局语料统计词对，而应该执行局部“增量修改”。
        affected_tokens = []
        for token_seq, freq in token_frequency_table.items():
            # 检查当前的 Token 序列内部是否存在需要合并的 best_pair
            has_pair = any(token_seq[i:i+2] == best_pair for i in range(len(token_seq) - 1))
            if has_pair:
                affected_tokens.append((token_seq, freq))

        # 4.5 针对受影响的词条序列，精确扣减和增加关联词对的全局频次计数
        for token_seq, freq in affected_tokens:
            # 减去旧序列对 pair_counts 的贡献：在旧序列被更新前，遍历它所包含的每个相邻词对，扣减其对应频次
            for i in range(len(token_seq) - 1):
                pair = (token_seq[i], token_seq[i+1])
                pair_counts[pair] -= freq
                if pair_counts[pair] <= 0:
                    del pair_counts[pair]  # 清理零频或负频项，维护字典大小

            # 调用辅助函数，在当前的 Token 序列中完成 best_pair 到 new_token_bytes 的替换
            new_token_seq = merge_token_sequence(token_seq, best_pair, new_token_bytes)

            # 加上新序列对 pair_counts 的贡献：遍历融合后的全新序列，将其产生的所有邻近词对频次累加回去
            for i in range(len(new_token_seq) - 1):
                pair = (new_token_seq[i], new_token_seq[i+1])
                pair_counts[pair] += freq

            # 同步更新全局词条状态表：删去旧序列，并在对应的位置加上（或创建）新序列的频次
            del token_frequency_table[token_seq]
            token_frequency_table[new_token_seq] += freq

    # ==========================================
    # 第 5 步: 序列化持久化存储
    # ==========================================
    # 保存词汇表到二进制文件，后续载入分词器时可直接映射 Token ID 与 字节串
    with open("vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)
    
    # 保存合并规则到二进制文件，这是进行文本 BPE Tokenize（编码）时必不可少的优先级依据
    with open("merges.pkl", "wb") as f:
        pickle.dump(merges, f)

    return vocab, merges  # 返回最终构建的词汇表字典及顺序合并规则列表

if __name__ == "__main__":
    special_tokens = ["<|endoftext|>"]
    # 示例调用：对 OWT 训练数据集运行 BPE 构建，目标词表大小设为 20000 
    vocab, merges = run_train_bpe("/home/niedongliang/workspace/cs336/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt", 20000, special_tokens)

    print(f"训练完成！当前词表实际大小: {len(vocab)}")
    print(f"前 5 条合并规则展示: {merges[:5]}")
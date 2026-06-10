import json
import time

# 导入待测的 BPE 训练入口函数
from .adapters import run_train_bpe
# FIXTURES_PATH: 测试基准数据文件夹路径
# gpt2_bytes_to_unicode: GPT-2 用于将原始字节映射到可见 Unicode 字符的函数（防止控制字符或空格在文本文件中引发格式错乱）
from .common import FIXTURES_PATH, gpt2_bytes_to_unicode


def test_train_bpe_speed():
    """
    确保 BPE 训练具有相对较高的效率。
    测试会在一个较小的数据集上统计训练耗时，
    如果训练时间超过 1.5 秒，则会抛出错误。
    这个时间限制实际上相当宽松：
    - 参考实现（reference implementation）在我的笔记本上运行只需要 0.38 秒。
    - 而课程里给出的 toy implementation（朴素实现）大约需要 3 秒。
    """
    # 获取测试用的英文语料库路径
    input_path = FIXTURES_PATH / "corpus.en"
    
    # 记录训练开始的时间戳
    start_time = time.time()
    
    # 运行 BPE 训练，设置词表大小为 500，并添加一个特殊令牌
    _, _ = run_train_bpe(
        input_path=input_path,
        vocab_size=500,
        special_tokens=["<|endoftext|>"],
    )
    
    # 记录训练结束的时间戳
    end_time = time.time()
    
    # 断言：整个训练耗时必须小于 1.5 秒（用以检查是否实现了算法优化，如使用高效的哈希表或堆结构）
    assert end_time - start_time < 1.5


def test_train_bpe():
    """
    验证 BPE 训练出的词表（Vocab）和合并规则（Merges）是否与标准的基准参考完全一致。
    """
    # 1. 运行当前实现的 BPE 算法进行训练
    input_path = FIXTURES_PATH / "corpus.en"
    vocab, merges = run_train_bpe(
        input_path=input_path,
        vocab_size=500,
        special_tokens=["<|endoftext|>"],
    )

    # 2. 获取预先保存好的基准参考（Reference）文件的路径
    reference_vocab_path = FIXTURES_PATH / "train-bpe-reference-vocab.json"
    reference_merges_path = FIXTURES_PATH / "train-bpe-reference-merges.txt"

    # 3. 解析并还原基准合并规则（Merges）
    # 关键点：GPT-2 在保存 merges.txt 时，使用的是映射后的可见字符（例如用 'Ġ' 代表空格）。
    # 为了和我们代码中纯粹的“原始字节（bytes）”比对，我们需要构建一个反向解码器。
    gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
    
    with open(reference_merges_path, encoding="utf-8") as f:
        # 读取参考合并规则，每行形如 "t h"，按空格切割成元组 ('t', 'h')
        gpt2_reference_merges = [tuple(line.rstrip().split(" ")) for line in f]
        
        # 将基准中的可见字符还原回原始的字节流（bytes），例如将 ('Ġ', 't') 还原为 (b' ', b't')
        reference_merges = [
            (
                bytes([gpt2_byte_decoder[token] for token in merge_token_1]),
                bytes([gpt2_byte_decoder[token] for token in merge_token_2]),
            )
            for merge_token_1, merge_token_2 in gpt2_reference_merges
        ]
        
    # 断言：当前代码学习到的合并顺序和规则，必须与官方基准完全相同
    assert merges == reference_merges

    # 4. 解析并还原基准词表（Vocab）
    with open(reference_vocab_path, encoding="utf-8") as f:
        # 基准词表是一个 JSON 字典，键是可见字符组成的字符串，值是对应的 Token ID 索引
        gpt2_reference_vocab = json.load(f)
        
        # 同样利用解码器，将词表中的字符还原成真实的字节串（bytes），保持 {ID: bytes} 的映射结构
        reference_vocab = {
            gpt2_vocab_index: bytes([gpt2_byte_decoder[token] for token in gpt2_vocab_item])
            for gpt2_vocab_item, gpt2_vocab_index in gpt2_reference_vocab.items()
        }
        
    # 5. 对比生成的词表与基准词表
    # 提示：由于字典在构建时的内部顺序或某些特定策略可能有异，这里不直接做 `vocab == reference_vocab` 的硬对比。
    # 而是分别确保两者的“所有 Token ID 集合”以及“所有字节内容集合”完全交配对齐。
    assert set(vocab.keys()) == set(reference_vocab.keys())
    assert set(vocab.values()) == set(reference_vocab.values())


def test_train_bpe_special_tokens(snapshot):
    """
    确保特殊令牌（Special Tokens）被正确地添加到词表中，
    并且绝不能与普通的文本字符发生错误的合并。
    """
    # 在稍大一点的数据集上进行训练，词表扩大到 1000
    input_path = FIXTURES_PATH / "tinystories_sample_5M.txt"
    vocab, merges = run_train_bpe(
        input_path=input_path,
        vocab_size=1000,
        special_tokens=["<|endoftext|>"],
    )

    # 1. 过滤掉词表中本就属于特殊令牌的项
    vocabs_without_specials = [word for word in vocab.values() if word != b"<|endoftext|>"]
    
    # 2. 安全安全性检查：
    # 普通词表中任何一个词的原始字节，都不应该包含 `b"<|"` 这样的片段。
    # 如果包含，说明 BPE 算法错误地破坏了特殊令牌的边界，或者在不该发生切分的地方把普通文本和特殊令牌的碎片缝合了。
    for word_bytes in vocabs_without_specials:
        assert b"<|" not in word_bytes

    # 3. 快照测试（Snapshot Testing）：
    # 将当前的词表键、值以及合并规则与上一次运行成功保留的“快照文件”进行对比，
    # 这样可以在不写死 hardcode 的情况下，敏锐捕捉到算法行为的任何微小变动。
    snapshot.assert_match(
        {
            "vocab_keys": set(vocab.keys()),
            "vocab_values": set(vocab.values()),
            "merges": merges,
        },
    )

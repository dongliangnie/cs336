from __future__ import annotations

import json
import os
import resource
import sys

import psutil
import pytest
import tiktoken

from .adapters import get_tokenizer
from .common import FIXTURES_PATH, gpt2_bytes_to_unicode

# 定义测试所需的 GPT-2 官方词汇表与合并规则文件的路径
VOCAB_PATH = FIXTURES_PATH / "gpt2_vocab.json"
MERGES_PATH = FIXTURES_PATH / "gpt2_merges.txt"


def memory_limit(max_mem):
    """
    一个用于动态限制测试函数运行期虚拟内存（Address Space）的装饰器。
    用于强行测试流式分词（encode_iterable）是否真正做到了内存友好。
    """
    def decorator(f):
        def wrapper(*args, **kwargs):
            # 获取当前测试进程的 PID
            process = psutil.Process(os.getpid())
            # 获取系统当前对进程虚拟内存（RLIMIT_AS）的软限制和硬限制
            prev_limits = resource.getrlimit(resource.RLIMIT_AS)
            
            # 核心控制：设置新的内存上限 = 当前进程已用常驻内存 (RSS) + 允许增加的最大内存 (max_mem)
            # -1 (即 resource.RLIM_INFINITY) 表示不对硬限制做额外修改
            resource.setrlimit(resource.RLIMIT_AS, (process.memory_info().rss + max_mem, -1))
            try:
                result = f(*args, **kwargs)
                return result
            finally:
                # 【关键防护】无论测试成功还是因超限触发异常崩溃，
                # 都在 finally 块中将内存限制恢复原状，避免污染和影响后续其他测试用例的执行
                resource.setrlimit(resource.RLIMIT_AS, prev_limits)

        return wrapper

    return decorator


def get_tokenizer_from_vocab_merges_path(
    vocab_path: str | os.PathLike,
    merges_path: str | os.PathLike,
    special_tokens: list[str] | None = None,
):
    """
    辅助函数：从 GPT-2 官方的 json 词表和 txt 合并规则文件中加载数据，
    并将其还原为通用的原始字节（bytes）格式，最后实例化并返回待测的分词器。
    """
    # GPT-2 官方为了避免控制字符或空格在文本展示中乱码，使用了一种特殊的映射（把单字节映射到不常用的 Unicode 字符）。
    # 这里反转字典：建立“映射后的 Unicode 字符 -> 原始单字节整数”的解码器
    gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
    
    # 1. 读取 GPT-2 官方词表
    with open(vocab_path) as vocab_f:
        gpt2_vocab = json.load(vocab_f)
        
    # 2. 读取 GPT-2 官方合并规则 (Merges)
    gpt2_bpe_merges = []
    with open(merges_path) as f:
        for line in f:
            cleaned_line = line.rstrip()
            # 过滤掉空行，并确保每行是由空格隔开的两个 Token 组合
            if cleaned_line and len(cleaned_line.split(" ")) == 2:
                gpt2_bpe_merges.append(tuple(cleaned_line.split(" ")))
                
    # 3. 核心还原逻辑：将官方词表中被重映射的字符串，逆向还原为干净的、学生易于处理的原始 bytes 序列
    # 例如：将词表项中的映射字符还原为 b'Hello'
    vocab = {
        gpt2_vocab_index: bytes([gpt2_byte_decoder[token] for token in gpt2_vocab_item])
        for gpt2_vocab_item, gpt2_vocab_index in gpt2_vocab.items()
    }
    
    # 4. 静态注入特殊符号：若传入的特殊符号（如 <|endoftext|>）不在基础词表中，将其序列化后追加至词表末尾
    if special_tokens:
        for special_token in special_tokens:
            byte_encoded_special_token = special_token.encode("utf-8")
            if byte_encoded_special_token not in set(vocab.values()):
                vocab[len(vocab)] = byte_encoded_special_token

    # 5. 同理，将官方的合并规则（字符串对）逆向解码还原为两组原始 bytes 构成的双元组
    merges = [
        (
            bytes([gpt2_byte_decoder[token] for token in merge_token_1]),
            bytes([gpt2_byte_decoder[token] for token in merge_token_2]),
        )
        for merge_token_1, merge_token_2 in gpt2_bpe_merges
    ]
    # 调用适配器接口，构造并返回分词器实例
    return get_tokenizer(vocab, merges, special_tokens)


# =====================================================================
# 基础边界测试（测试空字符串与单字符）
# =====================================================================

def test_roundtrip_empty():
    """测试空字符串编码后再解码，能否 100% 还原（可逆性测试）"""
    tokenizer = get_tokenizer_from_vocab_merges_path(vocab_path=VOCAB_PATH, merges_path=MERGES_PATH)
    test_string = ""
    encoded_ids = tokenizer.encode(test_string)
    decoded_string = tokenizer.decode(encoded_ids)
    assert test_string == decoded_string


def test_empty_matches_tiktoken():
    """测试空字符串的分词 ID 和解码行为是否与 OpenAI 官方的 tiktoken 完全一致"""
    reference_tokenizer = tiktoken.get_encoding("gpt2")
    tokenizer = get_tokenizer_from_vocab_merges_path(vocab_path=VOCAB_PATH, merges_path=MERGES_PATH)
    test_string = ""

    reference_ids = reference_tokenizer.encode(test_string)
    ids = tokenizer.encode(test_string)
    assert ids == reference_ids  # 断言生成的 Token ID 数组完全相等

    tokenized_string = [tokenizer.decode([x]) for x in ids]
    assert tokenized_string == []  # 空串切分后应为空列表

    assert tokenizer.decode(ids) == test_string
    assert reference_tokenizer.decode(reference_ids) == test_string


def test_roundtrip_single_character():
    """测试单个 ASCII 字符（如 's'）的编码与解码可逆性"""
    tokenizer = get_tokenizer_from_vocab_merges_path(vocab_path=VOCAB_PATH, merges_path=MERGES_PATH)
    test_string = "s"
    encoded_ids = tokenizer.encode(test_string)
    decoded_string = tokenizer.decode(encoded_ids)
    assert test_string == decoded_string


def test_single_character_matches_tiktoken():
    """断言单个 ASCII 字符的分词结果与官方 tiktoken 严丝合缝"""
    reference_tokenizer = tiktoken.get_encoding("gpt2")
    tokenizer = get_tokenizer_from_vocab_merges_path(vocab_path=VOCAB_PATH, merges_path=MERGES_PATH)
    test_string = "s"

    reference_ids = reference_tokenizer.encode(test_string)
    ids = tokenizer.encode(test_string)
    assert ids == reference_ids

    tokenized_string = [tokenizer.decode([x]) for x in ids]
    assert tokenized_string == ["s"]

    assert tokenizer.decode(ids) == test_string
    assert reference_tokenizer.decode(reference_ids) == test_string


def test_roundtrip_single_unicode_character():
    """测试多字节 Unicode 字符（例如 Emoji 表情 '🙃'，占用 4 个字节）的编解码鲁棒性"""
    tokenizer = get_tokenizer_from_vocab_merges_path(vocab_path=VOCAB_PATH, merges_path=MERGES_PATH)
    test_string = "🙃"
    encoded_ids = tokenizer.encode(test_string)
    decoded_string = tokenizer.decode(encoded_ids)
    assert test_string == decoded_string


def test_single_unicode_character_matches_tiktoken():
    """验证 Emoji 字符分词在多字节切分时是否与 tiktoken 一致（检查是否有非预期断裂）"""
    reference_tokenizer = tiktoken.get_encoding("gpt2")
    tokenizer = get_tokenizer_from_vocab_merges_path(vocab_path=VOCAB_PATH, merges_path=MERGES_PATH)
    test_string = "🙃"

    reference_ids = reference_tokenizer.encode(test_string)
    ids = tokenizer.encode(test_string)
    assert ids == reference_ids

    assert tokenizer.decode(ids) == test_string
    assert reference_tokenizer.decode(reference_ids) == test_string


# =====================================================================
# 进阶文本测试（常规英文短句与带有特殊符号、多国语言的文本）
# =====================================================================

def test_roundtrip_ascii_string():
    """常规 ASCII（纯英文带标点）长句的编解码一致性测试"""
    tokenizer = get_tokenizer_from_vocab_merges_path(vocab_path=VOCAB_PATH, merges_path=MERGES_PATH)
    test_string = "Hello, how are you?"
    encoded_ids = tokenizer.encode(test_string)
    decoded_string = tokenizer.decode(encoded_ids)
    assert test_string == decoded_string


def test_ascii_string_matches_tiktoken():
    """
    测试常规英文句子的切分片段。
    这里重点检测 GPT-2 标准下，前导空格是否正确地附着在单词头部（如 ' how', ' are'）。
    """
    reference_tokenizer = tiktoken.get_encoding("gpt2")
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH, merges_path=MERGES_PATH, special_tokens=["<|endoftext|>"]
    )
    test_string = "Hello, how are you?"

    reference_ids = reference_tokenizer.encode(test_string)
    ids = tokenizer.encode(test_string)
    # assert ids == reference_ids

    tokenized_string = [tokenizer.decode([x]) for x in ids]
    # 精确断言切分出的文本块，注意前导空格的位置
    assert tokenized_string == ["Hello", ",", " how", " are", " you", "?"]

    assert tokenizer.decode(ids) == test_string
    assert reference_tokenizer.decode(reference_ids) == test_string


def test_roundtrip_unicode_string():
    """测试混合了特殊变音符号（如 é, ò, ü）及扩展表情的复杂 Unicode 字符串可逆性"""
    tokenizer = get_tokenizer_from_vocab_merges_path(vocab_path=VOCAB_PATH, merges_path=MERGES_PATH)
    test_string = "Héllò hôw are ü? 🙃"
    encoded_ids = tokenizer.encode(test_string)
    decoded_string = tokenizer.decode(encoded_ids)
    assert test_string == decoded_string


def test_unicode_string_matches_tiktoken():
    """确保包含变音符号的多国语言文本在分词时仍能与 tiktoken 保持完美对齐"""
    reference_tokenizer = tiktoken.get_encoding("gpt2")
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH, merges_path=MERGES_PATH, special_tokens=["<|endoftext|>"]
    )
    test_string = "Héllò hôw are ü? 🙃"

    reference_ids = reference_tokenizer.encode(test_string)
    ids = tokenizer.encode(test_string)
    assert ids == reference_ids

    assert tokenizer.decode(ids) == test_string
    assert reference_tokenizer.decode(reference_ids) == test_string


# =====================================================================
# 特殊 Token 边界测试（核心考察贪婪匹配与割裂防护）
# =====================================================================

def test_roundtrip_unicode_string_with_special_tokens():
    """测试文本中嵌入了多个特殊符号（<|endoftext|>）时，编解码能否正确将它们独立识别而不被硬切开"""
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH, merges_path=MERGES_PATH, special_tokens=["<|endoftext|>"]
    )
    test_string = "Héllò hôw <|endoftext|><|endoftext|> are ü? 🙃<|endoftext|>"
    encoded_ids = tokenizer.encode(test_string)
    tokenized_string = [tokenizer.decode([x]) for x in encoded_ids]
    
    # 确保文本中独立提取出了 3 个完整的特殊标记，没有被拆分成 '<', '|', 'end' 等小元素
    assert tokenized_string.count("<|endoftext|>") == 3

    decoded_string = tokenizer.decode(encoded_ids)
    assert test_string == decoded_string


def test_unicode_string_with_special_tokens_matches_tiktoken():
    """将包含特殊标记的混合文本分词结果与 tiktoken 进行严格对照"""
    reference_tokenizer = tiktoken.get_encoding("gpt2")
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH, merges_path=MERGES_PATH, special_tokens=["<|endoftext|>"]
    )
    test_string = "Héllò hôw <|endoftext|><|endoftext|> are ü? 🙃<|endoftext|>"

    # 显式告知 tiktoken：允许解析文本里的 "<|endoftext|>" 为特殊标记而非纯文本
    reference_ids = reference_tokenizer.encode(test_string, allowed_special={"<|endoftext|>"})
    ids = tokenizer.encode(test_string)
    assert ids == reference_ids

    assert tokenizer.decode(ids) == test_string
    assert reference_tokenizer.decode(reference_ids) == test_string


def test_overlapping_special_tokens():
    """
    【高难边界】测试重叠/包含关系的特殊符号。
    定义了短标记 '<|endoftext|>' 和长标记 '<|endoftext|><|endoftext|>'。
    考察正则表达式/分词内核是否符合“最大匹配（Maximal Munch/Greedy Match）”原则，
    即优先匹配最长的连续合法特殊标记。
    """
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
        special_tokens=["<|endoftext|>", "<|endoftext|><|endoftext|>"],
    )
    test_string = "Hello, how <|endoftext|><|endoftext|> are you?<|endoftext|>"

    ids = tokenizer.encode(test_string)
    tokenized_string = [tokenizer.decode([x]) for x in ids]
    
    # 预期结果：中间的连续双重标记被合成为 1 个长 Token，结尾的单独标记被识别为 1 个短 Token
    assert tokenized_string.count("<|endoftext|>") == 1
    assert tokenized_string.count("<|endoftext|><|endoftext|>") == 1
    assert tokenizer.decode(ids) == test_string


# =====================================================================
# 真实文本语料大测试（读取 fixtures 目录下的各种典型长文本）
# =====================================================================

def test_address_roundtrip():
    """测试林肯葛底斯堡演说演讲稿（address.txt）的完整可逆性"""
    tokenizer = get_tokenizer_from_vocab_merges_path(vocab_path=VOCAB_PATH, merges_path=MERGES_PATH)
    with open(FIXTURES_PATH / "address.txt") as f:
        corpus_contents = f.read()

    ids = tokenizer.encode(corpus_contents)
    assert tokenizer.decode(ids) == corpus_contents


def test_address_matches_tiktoken():
    """对照标准历史英文语料，确保长文本 Token 序列完美一致"""
    reference_tokenizer = tiktoken.get_encoding("gpt2")
    tokenizer = get_tokenizer_from_vocab_merges_path(vocab_path=VOCAB_PATH, merges_path=MERGES_PATH)
    corpus_path = FIXTURES_PATH / "address.txt"
    with open(corpus_path) as f:
        corpus_contents = f.read()
    reference_ids = reference_tokenizer.encode(corpus_contents)
    ids = tokenizer.encode(corpus_contents)
    assert ids == reference_ids

    assert tokenizer.decode(ids) == corpus_contents
    assert reference_tokenizer.decode(reference_ids) == corpus_contents


def test_german_roundtrip():
    """测试包含大量德语特有连字及复合词（german.txt）的编解码闭环"""
    tokenizer = get_tokenizer_from_vocab_merges_path(vocab_path=VOCAB_PATH, merges_path=MERGES_PATH)
    with open(FIXTURES_PATH / "german.txt") as f:
        corpus_contents = f.read()

    ids = tokenizer.encode(corpus_contents)
    assert tokenizer.decode(ids) == corpus_contents


def test_german_matches_tiktoken():
    """核对德语语料切分是否符合官方 GPT-2 的合并习惯"""
    reference_tokenizer = tiktoken.get_encoding("gpt2")
    tokenizer = get_tokenizer_from_vocab_merges_path(vocab_path=VOCAB_PATH, merges_path=MERGES_PATH)
    corpus_path = FIXTURES_PATH / "german.txt"
    with open(corpus_path) as f:
        corpus_contents = f.read()
    reference_ids = reference_tokenizer.encode(corpus_contents)
    ids = tokenizer.encode(corpus_contents)
    assert ids == reference_ids

    assert tokenizer.decode(ids) == corpus_contents
    assert reference_tokenizer.decode(reference_ids) == corpus_contents


def test_tinystories_sample_roundtrip():
    """测试微型故事数据集样例（tinystories_sample.txt）的编解码闭环"""
    tokenizer = get_tokenizer_from_vocab_merges_path(vocab_path=VOCAB_PATH, merges_path=MERGES_PATH)
    with open(FIXTURES_PATH / "tinystories_sample.txt") as f:
        corpus_contents = f.read()

    ids = tokenizer.encode(corpus_contents)
    assert tokenizer.decode(ids) == corpus_contents


def test_tinystories_matches_tiktoken():
    """核对微型故事样例的分词序列是否全量契合官方规范"""
    reference_tokenizer = tiktoken.get_encoding("gpt2")
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH, merges_path=MERGES_PATH, special_tokens=["<|endoftext|>"]
    )
    corpus_path = FIXTURES_PATH / "tinystories_sample.txt"
    with open(corpus_path) as f:
        corpus_contents = f.read()
    reference_ids = reference_tokenizer.encode(corpus_contents, allowed_special={"<|endoftext|>"})
    ids = tokenizer.encode(corpus_contents)
    assert ids == reference_ids

    assert tokenizer.decode(ids) == corpus_contents
    assert reference_tokenizer.decode(reference_ids) == corpus_contents


# =====================================================================
# 回归与极端空白边缘情况测试（Regression Tests）
# =====================================================================

def test_encode_special_token_trailing_newlines():
    """回归测试：考察当特殊标记紧随尾部换行符时，预分词正则表达式（PAT）会不会发生误切或吞字"""
    reference_tokenizer = tiktoken.get_encoding("gpt2")
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH, merges_path=MERGES_PATH, special_tokens=["<|endoftext|>"]
    )
    corpus_path = FIXTURES_PATH / "special_token_trailing_newlines.txt"
    with open(corpus_path) as f:
        corpus_contents = f.read()
    reference_ids = reference_tokenizer.encode(corpus_contents, allowed_special={"<|endoftext|>"})
    ids = tokenizer.encode(corpus_contents)
    assert ids == reference_ids

    assert tokenizer.decode(ids) == corpus_contents
    assert reference_tokenizer.decode(reference_ids) == corpus_contents


def test_encode_special_token_double_newline_non_whitespace():
    """回归测试：考察当双换行符后紧跟非空白字符并与特殊标记组合时，分词处理的健壮性"""
    reference_tokenizer = tiktoken.get_encoding("gpt2")
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH, merges_path=MERGES_PATH, special_tokens=["<|endoftext|>"]
    )
    corpus_path = FIXTURES_PATH / "special_token_double_newlines_non_whitespace.txt"
    with open(corpus_path) as f:
        corpus_contents = f.read()
    reference_ids = reference_tokenizer.encode(corpus_contents, allowed_special={"<|endoftext|>"})
    ids = tokenizer.encode(corpus_contents)
    assert ids == reference_ids

    assert tokenizer.decode(ids) == corpus_contents
    assert reference_tokenizer.decode(reference_ids) == corpus_contents


# =====================================================================
# 生成器/流式分词与高阶内存效率测试（Memory Efficiency Tests）
# =====================================================================

def test_encode_iterable_tinystories_sample_roundtrip():
    """测试高级流式分词接口 `encode_iterable`（传入文件句柄/可迭代对象），解码后仍能完好还原整个大文件"""
    tokenizer = get_tokenizer_from_vocab_merges_path(vocab_path=VOCAB_PATH, merges_path=MERGES_PATH)
    all_ids = []
    # 逐块或逐行读取文本，并流式产出 ID，不一次性将全量文本转为巨型字符串
    with open(FIXTURES_PATH / "tinystories_sample.txt") as f:
        for _id in tokenizer.encode_iterable(f):
            all_ids.append(_id)
    with open(FIXTURES_PATH / "tinystories_sample.txt") as f:
        corpus_contents = f.read()
    assert tokenizer.decode(all_ids) == corpus_contents


def test_encode_iterable_tinystories_matches_tiktoken():
    """验证流式分词接口产生的结果，与全量拉取编码在结果上等价且完全匹配官方标准"""
    reference_tokenizer = tiktoken.get_encoding("gpt2")
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH, merges_path=MERGES_PATH, special_tokens=["<|endoftext|>"]
    )
    corpus_path = FIXTURES_PATH / "tinystories_sample.txt"
    with open(corpus_path) as f:
        corpus_contents = f.read()
    reference_ids = reference_tokenizer.encode(corpus_contents, allowed_special={"<|endoftext|>"})
    all_ids = []
    with open(FIXTURES_PATH / "tinystories_sample.txt") as f:
        for _id in tokenizer.encode_iterable(f):
            all_ids.append(_id)
    assert all_ids == reference_ids

    assert tokenizer.decode(all_ids) == corpus_contents
    assert reference_tokenizer.decode(reference_ids) == corpus_contents


@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="rlimit 对非 Linux 系统（如 macOS/Windows）的支持很不稳定或不支持。",
)
def test_encode_iterable_memory_usage():
    """
    【内存极限测试 1】：处理一个高达 5MB 的较大型文本文件。
    由于使用的是高效的流式分词（_encode_iterable），它在运行期间的虚拟内存增量
    被控制在 1MB 的超小范围内。
    预期结果：测试顺利通过（PASS），证明实现了优秀的惰性求值/流式流转。
    """
    tokenizer = get_tokenizer_from_vocab_merges_path(vocab_path=VOCAB_PATH, merges_path=MERGES_PATH)
    with open(FIXTURES_PATH / "tinystories_sample_5M.txt") as f:
        ids = []
        for _id in _encode_iterable(tokenizer, f):
            ids.append(_id)


@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="rlimit 对非 Linux 系统（如 macOS/Windows）的支持很不稳定或不支持。",
)
@pytest.mark.xfail(reason="预期的失败：常规 Tokenizer.encode 一次性吃下整个大文本，内存必然会超出 1MB 软硬限制。")
def test_encode_memory_usage():
    """
    【内存极限测试 2】：对比组。
    常规的 `encode` 函数必须先通过 `f.read()` 将 5MB 的纯文本塞进内存，随后构建庞大的对象。
    我们给它加了 `@pytest.mark.xfail` 标记，表示【预期它会失败】。
    如果在 1MB 的极限限制下它抛出了超限崩溃，说明这个测试本身是完全符合逻辑的。
    """
    tokenizer = get_tokenizer_from_vocab_merges_path(vocab_path=VOCAB_PATH, merges_path=MERGES_PATH)
    with open(FIXTURES_PATH / "tinystories_sample_5M.txt") as f:
        contents = f.read()
        _ = _encode(tokenizer, contents)


@memory_limit(int(1e6))
def _encode_iterable(tokenizer, iterable):
    """
    流式编码辅助函数。通过装饰器将其整个生命周期的虚拟内存增量限制在 1MB ($10^6$ 字节) 以内。
    采用 `yield from` 实现零拷贝级的数据发生器传递。
    """
    yield from tokenizer.encode_iterable(iterable)


@memory_limit(int(1e6))
def _encode(tokenizer, text):
    """
    常规编码辅助函数。通过装饰器将其整个生命周期的虚拟内存增量限制在 1MB ($10^6$ 字节) 以内。
    """
    return tokenizer.encode(text)

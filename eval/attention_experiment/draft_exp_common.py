#!/usr/bin/env python3
"""3 个实验的公共基础设施: 加载模型/词表, 强制末尾锚点, 逐层注意力采集, 关键token跨度定位。

复用于 exp1_length_position / exp2_distractors / exp3_update。
"""
import re
from pathlib import Path

import numpy as np
import torch
import torch_npu  # noqa: F401

MODEL = "/home/y50063564/Qwen3-4B"
CKPT = "/home/y50063564/checkpoint_best"
TOKEN_FREQ = "/home/y50063564/data/open_perfectblend_qwen3_4b_700k/token_freq.pt"
TARGET_LAYERS = [1, 9, 17, 25, 33]
DRAFT_VOCAB = 32000


def load_models(device: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    verifier = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, attn_implementation="eager"
    )
    verifier.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    from speculators.model import SpeculatorModel
    from speculators.models.attention import create_float_mask
    import speculators.models.dflash.core as core_mod

    draft = SpeculatorModel.from_pretrained(CKPT, trust_remote_code=True, verifier=MODEL)
    draft.to(device).eval()
    draft._create_mask_fn = create_float_mask
    for layer in draft.layers:
        layer.self_attn.config._attn_implementation = "eager"

    orig_select = core_mod.select_anchors
    def forced_select(loss_mask, num_anchors, block_size):
        anchors, valid = orig_select(loss_mask, num_anchors, block_size)
        anchors = anchors.clone(); valid = valid.clone()
        anchors[-1] = loss_mask.shape[1] - 1
        valid[-1] = True
        return anchors, valid
    core_mod.select_anchors = forced_select

    orig_build = draft._build_attention_mask
    def wrapper(loss_mask, max_anchors, document_ids, device):
        out = orig_build(loss_mask, max_anchors, document_ids, device)
        draft._last_anchors = out[2].cpu()
        return out
    draft._build_attention_mask = wrapper
    return verifier, tokenizer, draft


def build_vocab(tokenizer) -> list[int]:
    tf = torch.load(TOKEN_FREQ, map_location="cpu", weights_only=True)
    sel = sorted(tf, key=lambda t: (-tf[t], t))[:DRAFT_VOCAB]
    sel.sort()
    return sel


def run_forward(draft, aux, ids, loss_mask, vlast, doc_ids):
    """返回 {layer_attn:[L], fc, logits}。强制末尾锚点已生效。"""
    from speculators.models.dflash.core import DFlashDraftModel
    cap = {}
    handles = []
    for i in range(5):
        def mk(idx):
            def h(mod, inp, out):
                cap.setdefault("layer_attn", {})[idx] = out[1].detach().float().cpu()
            return h
        handles.append(draft.layers[i].self_attn.register_forward_hook(mk(i)))
    orig_nf = draft.hidden_norm.forward
    def nf(inp):
        o = orig_nf(inp)
        cap["fc"] = o.detach().float().cpu()
        return o
    draft.hidden_norm.forward = nf
    bb = DFlashDraftModel._backbone_forward
    def wbb(self, *a, **kw):
        r = bb(self, *a, **kw)
        cap["logits"] = r[1].detach().float().cpu()
        return r
    DFlashDraftModel._backbone_forward = wbb
    with torch.no_grad():
        draft(hidden_states=aux, input_ids=ids, loss_mask=loss_mask,
              verifier_last_hidden_states=vlast, document_ids=doc_ids, max_anchors=256)
    DFlashDraftModel._backbone_forward = bb
    draft.hidden_norm.forward = orig_nf
    for h in handles:
        h.remove()
    return cap


def answer_slot(draft):
    return (draft._last_anchors.shape[0] - 1) * draft.block_size + 1


def find_span(tokens: list[str], needle: str, max_span: int = 40) -> list[int]:
    """找连续 token 跨度, 其拼接文本含 needle。"""
    norm = re.sub(r"\s+", "", needle.lower())
    texts = [re.sub(r"\s+", "", t.lower()) for t in tokens]
    best = None
    for start in range(len(tokens)):
        acc = ""
        for end in range(start, min(len(tokens), start + max_span)):
            acc += texts[end]
            if norm in acc:
                if best is None or (end - start) < (best[1] - best[0]):
                    best = (start, end)
                break
            if len(acc) > len(norm) + 40:
                break
    return [] if best is None else list(range(best[0], best[1] + 1))


def p_of_draft_id(logits, slot, draft_id, vocab):
    """logits[0, slot] 中 draft_id 的 softmax 概率(draft_id 可能多子词→只查单子词)。"""
    if draft_id is None or draft_id >= vocab.shape[0]:
        return None
    return torch.softmax(logits[0, slot], dim=-1)[draft_id].item()


def first_subword_draft_id(tokenizer, vocab, answer_word: str) -> int | None:
    """答案词第一个子词的 verifier id → draft id(需单子词)。"""
    aid = tokenizer.encode(f" {answer_word}")[-1] if tokenizer.encode(f" {answer_word}") else None
    # 取答案词第一个完整子词: 用 encode 找第一个 id
    ids = tokenizer.encode(f" {answer_word}")
    # 第一个 id 可能不是词的开始(如空格), 用 find_span 语义: 直接取答案词文本在 vocab 的索引
    # 简化: 若答案词单 token 在词表, 直接返回; 否则取 encode 的第一个 id
    for txt in (f" {answer_word}", answer_word):
        one = tokenizer.encode(txt)
        if len(one) == 1 and one[0] < len(vocab):
            try:
                return vocab.index(one[0])
            except ValueError:
                pass
    # 多子词: 取第一个子词 id
    first = ids[0]
    try:
        return vocab.index(first)
    except ValueError:
        return None


def prepare_inputs(verifier, tokenizer, prompt, device):
    ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    T = ids.shape[1]
    with torch.no_grad():
        hs = verifier(ids, output_hidden_states=True).hidden_states
    aux = torch.cat([hs[i].float() for i in TARGET_LAYERS], dim=-1).to(torch.bfloat16)
    vlast = hs[-1]
    loss_mask = torch.ones_like(ids).float()
    doc = torch.zeros_like(ids)
    toks = [tokenizer.decode([i]) for i in ids[0].tolist()]
    return ids, aux, vlast, loss_mask, doc, toks, T


# ===== 中性填充句池(多样、不重复) =====
FILLER_POOL = [
    "The old harbor once handled thousands of ships every year before the canal was built.",
    "Chemists often store volatile liquids in sealed glass containers away from direct sunlight.",
    "A researcher on the expedition recorded the temperature and humidity every two hours.",
    "The medieval bridge still stands across the river, connecting the two halves of the town.",
    "Each autumn, the maple trees along the avenue turn a deep shade of crimson.",
    "The orchestra rehearsed the new symphony for several weeks before the public premiere.",
    "Mountain climbers usually carry ropes, crampons, and a reliable weather forecast.",
    "The small museum displays fossils collected from the nearby limestone cliffs.",
    "Linguists trace the origin of many English words back to the ancient Germanic tribes.",
    "A farmer in the valley grows wheat, barley, and a few rows of sunflowers.",
    "The committee voted unanimously to postpone the meeting until the budget was approved.",
    "Astronomers use large telescopes to study galaxies that are billions of light-years away.",
    "The old textbook contained detailed diagrams of the human circulatory system.",
    "Every morning the librarian opens the reading room and arranges the newspapers.",
    "The bakery on the corner sells fresh bread, pastries, and a strong cup of coffee.",
    "During the winter, the lake freezes over and children skate on its surface.",
    "The engineer inspected the bridge supports and found a small crack near the base.",
    "Historians disagree about the exact date when the ancient city was first settled.",
    "A group of students designed a simple robot that could follow a painted line.",
    "The gardener prunes the rose bushes in early spring to encourage new growth.",
    "Sailors navigate by the stars when the instruments on board fail to respond.",
    "The scientist carefully labeled every sample before placing it in the freezer.",
    "Tourists visit the old cathedral to admire its stained glass windows.",
    "The chef experimented with new spices to improve the flavor of the soup.",
    "Rainwater collected in the reservoir supplies drinking water to the city below.",
    "The archaeologist brushed the dust from the ancient tablet to read the inscription.",
    "Writers often keep a notebook by their bed to capture ideas that come at night.",
    "The mechanic checked the engine oil level before starting the long journey.",
    "A flock of birds gathered on the telephone wires as the evening approached.",
    "The teacher wrote the assignment on the board and answered questions from the class.",
]

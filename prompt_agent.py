# prompt_agent.py
# Python 3.9 compatible

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from llm_utils import XiaoHuAPIClient


# =========================
# 手册骨架（用于三智能体内部推理，不直接喂给 OpenPrompt）
# =========================
BASE_PLAYBOOK = (
    '你是一名智能客服助手，负责辅助判断电商直播弹幕是否表达了用户购买兴趣。'
    '目标弹幕是：“{"placeholder": "text_a"}”,'
    '用户表达的含义具有{"mask"}倾向。'
)

# 注意：如果你未来真有 text_b（解释列）再打开
BASE_PLAYBOOK_WITH_B = (
    '你是一名智能客服助手，负责辅助判断电商直播弹幕是否表达了用户购买兴趣。'
    '目标弹幕是：“{"placeholder": "text_a"}”,'
    '对这条弹幕的解释是：“{"placeholder": "text_b"}”。'
    '用户表达的含义具有{"mask"}倾向。'
)

_PH_A = '{"placeholder":"text_a"}'
_PH_B = '{"placeholder":"text_b"}'
_MASK = '{"mask"}'

_PH_A_SP = '{"placeholder": "text_a"}'
_PH_B_SP = '{"placeholder": "text_b"}'


def _normalize_placeholders(text: str) -> str:
    if text is None:
        return ""
    return (
        text.replace(_PH_A_SP, _PH_A)
            .replace(_PH_B_SP, _PH_B)
            .replace('{"mask"}', _MASK)
    )


def _sanitize_curly_braces_keep_placeholders(text: str) -> str:
    """
    OpenPrompt ManualTemplate 会把 {...} 当作混合 token 标记。
    我们只允许 3 个合法标记：
      {"placeholder":"text_a"}, {"placeholder":"text_b"(可选)}, {"mask"}
    其他所有 { } 都替换为中文括号，避免解析炸。
    """
    text = _normalize_placeholders(text)

    sentinel_a = "__PH_TEXT_A__"
    sentinel_b = "__PH_TEXT_B__"
    sentinel_m = "__PH_MASK__"

    text = text.replace(_PH_A, sentinel_a).replace(_PH_B, sentinel_b).replace(_MASK, sentinel_m)

    text = text.replace("{", "（").replace("}", "）")

    text = text.replace(sentinel_a, _PH_A).replace(sentinel_b, _PH_B).replace(sentinel_m, _MASK)
    return text


def export_openprompt_template(use_text_b: bool = False) -> str:
    """
    这是最终写入 auto_agent_template.txt 的模板：
    - 必须单行
    - 只能出现 OpenPrompt 合法的 2/3 个花括号标记
    """
    if use_text_b:
        template = f'弹幕：{_PH_A}。解释：{_PH_B}。该弹幕更偏向{_MASK}。'
    else:
        template = f'弹幕：{_PH_A}。该弹幕更偏向{_MASK}。'

    template = _sanitize_curly_braces_keep_placeholders(template)
    template = re.sub(r"\s+", " ", template).strip()
    return template


def label_to_str(y: Any) -> str:
    s = str(y).strip()
    if s in ("1", "true", "True", "YES", "yes"):
        return "1"
    return "0"


def build_question(text_a: str) -> str:
    return (
        '在社交平台的电商直播中，用户发送的弹幕是：“{}”。'
        '请根据手册判断这条弹幕是否属于【1-推荐相关弹幕，可推荐】还是【0-闲聊/不推荐相关弹幕，不可推荐】。'
    ).format(text_a)


def format_generator_prompt(playbook: str, question: str) -> str:
    return (
        "你是一名电商平台弹幕分析专家。请参考手册回答问题。\n"
        "输出必须是 JSON，字段：reasoning, bullet_ids, final_answer(只输出1或0)。\n"
        "手册：\n"
        "{}\n"
        "问题：{}\n"
        "JSON格式：\n"
        "{{\n"
        "  \"reasoning\": \"...\",\n"
        "  \"bullet_ids\": [\"1.1\"],\n"
        "  \"final_answer\": \"1\"\n"
        "}}\n"
    ).format(playbook, question)


def format_reflector_prompt(
    question: str,
    model_trace_json: Dict[str, Any],
    pred_label: str,
    true_label: str,
    playbook_used: str,
) -> str:
    pred_cn = "可推荐" if pred_label == "1" else "不可推荐"
    true_cn = "可推荐" if true_label == "1" else "不可推荐"

    return (
        "你是反思器。比较预测与真实标签，指出错误原因，并给出可用于改进手册的要点。\n"
        "输出JSON字段：reasoning,error_identification,root_cause_analysis,correct_approach,key_insight,"
        "bullet_tags,modify_bullet,new_bullet,delete_bullet。\n\n"
        "问题：{}\n"
        "轨迹：{}\n"
        "预测：{}\n"
        "真实：{}\n"
        "手册：{}\n"
        "JSON：{{...}}\n"
    ).format(
        question,
        json.dumps(model_trace_json, ensure_ascii=False),
        pred_cn,
        true_cn,
        playbook_used,
    )


def format_integrator_prompt(reflections: List[Dict[str, Any]], current_playbook: str) -> str:
    reflections_str = json.dumps(reflections, ensure_ascii=False, indent=2)
    return (
        "你是整合器。根据反思更新手册。只输出修订后的手册正文，不要输出JSON，不要加外层花括号。\n\n"
        "反思：\n{}\n\n"
        "当前手册：\n{}\n"
    ).format(reflections_str, current_playbook)


def ensure_playbook_has_placeholders(playbook: str, use_text_b: bool) -> str:
    playbook = _normalize_placeholders(playbook)

    # 兜底：如果模型把占位符弄丢了，就用 base
    need_a = _PH_A in playbook
    need_m = _MASK in playbook
    need_b = (not use_text_b) or (_PH_B in playbook)

    if need_a and need_m and need_b:
        return playbook

    base = _normalize_placeholders(BASE_PLAYBOOK_WITH_B if use_text_b else BASE_PLAYBOOK)
    if playbook.strip():
        return base + "\n" + playbook
    return base


@dataclass
class GeneratorAgent:
    client: XiaoHuAPIClient

    def run(self, playbook: str, question: str) -> Dict[str, Any]:
        prompt = format_generator_prompt(playbook, question)
        return self.client.chat_json(
            [
                {"role": "system", "content": "只输出合法JSON。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )


@dataclass
class ReflectorAgent:
    client: XiaoHuAPIClient

    def run(
        self,
        question: str,
        model_trace_json: Dict[str, Any],
        pred_label: str,
        true_label: str,
        playbook_used: str,
    ) -> Dict[str, Any]:
        prompt = format_reflector_prompt(question, model_trace_json, pred_label, true_label, playbook_used)
        return self.client.chat_json(
            [
                {"role": "system", "content": "只输出合法JSON。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )


@dataclass
class IntegratorAgent:
    client: XiaoHuAPIClient

    def run(self, reflections: List[Dict[str, Any]], current_playbook: str, use_text_b: bool) -> str:
        prompt = format_integrator_prompt(reflections, current_playbook)
        text = self.client.chat_text(
            [
                {"role": "system", "content": "只输出修订后的手册正文，不要外层花括号，不要JSON。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        ).strip()

        # 防御：如果开头/结尾有孤立 { } 行，删掉
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if lines and lines[0] == "{":
            lines = lines[1:]
        if lines and lines[-1] == "}":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

        return ensure_playbook_has_placeholders(text, use_text_b=use_text_b)


@dataclass
class PromptOptimizationAgent:
    client: XiaoHuAPIClient
    text_a_col: int = 2
    label_col: int = 3
    text_b_col: Optional[int] = None

    def __post_init__(self) -> None:
        self.use_text_b = self.text_b_col is not None
        self.base_playbook = _normalize_placeholders(BASE_PLAYBOOK_WITH_B if self.use_text_b else BASE_PLAYBOOK)

        self.generator = GeneratorAgent(self.client)
        self.reflector = ReflectorAgent(self.client)
        self.integrator = IntegratorAgent(self.client)

    def load_data(self, csv_path: str) -> pd.DataFrame:
        df = pd.read_csv(csv_path, header=None)
        if df.shape[1] <= max(self.text_a_col, self.label_col):
            raise ValueError(
                f"CSV列数不足：{df.shape[1]}，需要至少到 text_a_col={self.text_a_col}, label_col={self.label_col}"
            )
        return df

    def sample_balanced(self, df: pd.DataFrame, k_per_class: int = 3, seed: int = 123) -> List[Tuple[str, str, str]]:
        df = df.copy()
        df["__y__"] = df.iloc[:, self.label_col].apply(label_to_str)
        pos = df[df["__y__"] == "1"]
        neg = df[df["__y__"] == "0"]

        k_pos = min(len(pos), k_per_class)
        k_neg = min(len(neg), k_per_class)

        pos_s = pos.sample(n=k_pos, random_state=seed) if k_pos > 0 else pos
        neg_s = neg.sample(n=k_neg, random_state=seed) if k_neg > 0 else neg

        mixed = pd.concat([pos_s, neg_s], axis=0).sample(frac=1.0, random_state=seed)

        samples: List[Tuple[str, str, str]] = []
        for _, r in mixed.iterrows():
            text_a = str(r.iloc[self.text_a_col])
            text_b = ""
            if self.use_text_b and self.text_b_col is not None and self.text_b_col < len(r):
                text_b = str(r.iloc[self.text_b_col])
            y = str(r["__y__"])
            samples.append((text_a, text_b, y))
        return samples

    def run_optimization(self, train_csv_path: str, max_iters: int = 3, seed: int = 123, k_per_class: int = 3) -> str:
        df = self.load_data(train_csv_path)
        samples = self.sample_balanced(df, k_per_class=k_per_class, seed=seed)

        playbook = ensure_playbook_has_placeholders(self.base_playbook, use_text_b=self.use_text_b)

        for it in range(max_iters):
            round_reflections: List[Dict[str, Any]] = []

            for (text_a, _text_b, y_true) in samples:
                question = build_question(text_a)

                gen_out = self.generator.run(playbook, question)
                pred = label_to_str(gen_out.get("final_answer", "0"))
                trace = {
                    "reasoning": gen_out.get("reasoning", ""),
                    "bullet_ids": gen_out.get("bullet_ids", []),
                    "final_answer": pred,
                }

                ref_out = self.reflector.run(
                    question=question,
                    model_trace_json=trace,
                    pred_label=pred,
                    true_label=y_true,
                    playbook_used=playbook,
                )
                ref_out["_meta"] = {"iter": it, "text_a": text_a, "pred": pred, "true": y_true}
                round_reflections.append(ref_out)

            playbook = self.integrator.run(round_reflections, playbook, use_text_b=self.use_text_b)

        return playbook

    def export_template(self, playbook: str) -> str:
        # 最终落盘给 OpenPrompt 的一行模板
        return export_openprompt_template(use_text_b=self.use_text_b)

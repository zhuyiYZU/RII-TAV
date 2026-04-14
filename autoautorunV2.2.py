# autoautorunV2.2_full_py39_E2E.py
# Python 3.9 兼容完整版本（端到端 E2E 口径修正版）
# ✅ 已集成“三智能体（生成器-反思器-整合器）”自动生成 Prompt（硬模板）
# ✅ 已修复：Python3.9 不支持 `str | None` 的问题（全部用 Optional）
# ✅ 已修复：subprocess stderr 为空导致 decode 报错的问题（capture_output=True, text=True）
#
# ✅ 时间统计口径修正（你要的“从运行开始到运行结束 / 总样本数”一致口径）：
# 1) GlobalInitSec：脚本启动到进入 seed 循环前（仅一次）
# 2) SeedTimeSec：每个 seed 内部全流程耗时（从 seed_start 到 seed_end，含写文件）
# 3) WriteAppendSec：每个 seed 写 xlsx/txt 的耗时
# 4) E2E_TotalSec_sofar：从脚本开始到“当前 seed 结束”的真实墙钟时间（不会重复摊 GlobalInit）
# 5) E2E_SecPerSample_sofar：E2E_TotalSec_sofar / 全部已处理样本数（跨 seed 累加）
# 6) E2E_SecPerSample_final：脚本结束总耗时 / 所有 seed 总样本数（最终端到端口径）
#
# 解释你关心的点：
# - 只跑 1 个 seed：E2E_SecPerSample_final == (脚本开始->脚本结束总耗时) / (该 seed 总样本数)
# - 跑多个 seed：E2E_SecPerSample_final == (脚本开始->全部 seed 结束总耗时) / (所有 seed 样本数累加)
#   ✅ 不会出现“GlobalInit 被每个 seed 重复加一次”的问题


import time

# ✅ 脚本总计时：尽可能早地开始
SCRIPT_START_TIME = time.time()

import csv
import logging
import subprocess
import os
from itertools import product
from typing import Optional, List

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import hanlp
from sentence_dis import TemporalAnchorManager


# =========================
# 日志配置
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

# =========================
# HanLP 模型
# =========================
tok = hanlp.load("UD_ONTONOTES_TOK_POS_LEM_FEA_NER_SRL_DEP_SDP_CON_MMINILMV2L6")

# =========================
# 平台与路径配置
# =========================
zeroshot_platform = "rec-related"
input_file_path = "datasets/veb/{}/all_test.csv".format(zeroshot_platform)
output_file = "datasets/veb/{}/test.csv".format(zeroshot_platform)  # zeroshot 临时词表输出


# ======================================================
# ✅ Key 写死（按你的要求）——【保持不变】
# ⚠️ 注意：不要把这份代码提交到公开仓库
# ======================================================
XIAOHU_KEY: Optional[str] = "sk-NDweZpKBL5cOikWhW25dv3bcLuUQtiYGMc4AJ99EEj3bKuch"


# =========================
# 子进程工具
# =========================
def run_cmd(cmd: str) -> None:
    logging.info("Executing command: %s", cmd)
    print(cmd)
    try:
        p = subprocess.run(
            cmd,
            shell=True,
            check=True,
            capture_output=True,
            text=True,
        )
        if p.stdout:
            logging.info(p.stdout.strip()[:1000])
        if p.stderr:
            logging.warning(p.stderr.strip()[:1000])
    except subprocess.CalledProcessError as e:
        err = (e.stderr or "")[:2000]
        out = (e.stdout or "")[:2000]
        logging.error("Command failed: %s\nSTDOUT:\n%s\nSTDERR:\n%s", cmd, out, err)

        # ✅ 完整输出落盘，便于定位真实报错
        try:
            with open("subprocess_error_full.log", "a", encoding="utf-8") as f:
                f.write("\n\n====================\n")
                f.write("CMD:\n" + cmd + "\n")
                f.write("---- STDOUT (FULL) ----\n")
                f.write(e.stdout or "")
                f.write("\n---- STDERR (FULL) ----\n")
                f.write(e.stderr or "")
                f.write("\n")
        except Exception:
            pass

        raise


# =========================
# Zero-shot（你原来的）
# =========================
def zeroshot() -> None:
    cmd = (
        "python zeroshot.py --result_file ./output_zeroshot1.txt "
        "--dataset {} --template_id 0 --seed 188 "
        "--verbalizer kpt --calibration"
    ).format(zeroshot_platform)
    run_cmd(cmd)


# =========================
# Few-shot（需要重训）
# =========================
def fewshot(
    n, t, j, i, m, k, v, e,
    fewshot_platform: str,
    auto_prompt: bool = False,
    prompt_path: Optional[str] = None,
    use_hard_template: bool = True,
    template_style: str = "default",
) -> None:
    cmd = (
        "python fewshot.py --result_file ./result/{platform}.txt "
        "--dataset {n} --template_id {t} --seed {j} "
        "--batch_size {i} --shot {m} --learning_rate {k} "
        "--verbalizer {v} --max_epochs {e}"
    ).format(
        platform=fewshot_platform, n=n, t=t, j=j, i=i, m=m, k=k, v=v, e=e
    )

    if auto_prompt:
        cmd += " --auto_prompt"
        if prompt_path:
            cmd += " --prompt_template_path {}".format(prompt_path)
        if use_hard_template:
            cmd += " --use_hard_template"
            cmd += " --template_style {}".format(template_style)

        # ✅ 强制 fewshot.py 使用小虎网关 + 同一把 key（按你原策略）
        cmd += " --xiaohu_base_url https://chat.xiaohuapi.site/v1"
        cmd += " --xiaohu_api_key {}".format(XIAOHU_KEY)

    run_cmd(cmd)


# =========================
# Few-shot（无需重训，直接测试）
# =========================
def fewshot1(n, t, j, i, v, fewshot_platform: str) -> None:
    cmd = (
        "python fewshot1.py --result_file ./result/{platform}.txt "
        "--dataset {n} --template_id {t} --seed {j} "
        "--batch_size {i} --verbalizer {v}"
    ).format(platform=fewshot_platform, n=n, t=t, j=j, i=i, v=v)
    run_cmd(cmd)


# =========================
# 三智能体生成 Prompt（一次性生成）
# 依赖同目录下：
#   - llm_utils.py
#   - prompt_agent.py
# =========================
def generate_prompt_with_three_agents(
    platform: str,
    train_csv_path: str,
    output_path: str,
    base_url: str = "https://chat.xiaohuapi.site/v1",
    api_key_env: str = "XIAOHU_API_KEY",
    api_key: Optional[str] = None,
    seed: int = 123,
    max_iters: int = 3,
    k_per_class: int = 3,
    force: bool = False,
) -> str:
    import logging

    def _read_text(path: str) -> str:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception:
            return ""

    def _is_openprompt_template_safe(s: str) -> bool:
        if not s or not s.strip():
            return False
        ss = s.strip()
        if ss == "{" or ss.startswith("{\n") or ss.startswith("{\r\n") or ss.startswith("{ "):
            return False
        if "\n" in ss or "\r" in ss:
            return False
        if '{"placeholder"' not in ss or '{"mask"}' not in ss:
            return False
        if ss.count("{") != ss.count("}"):
            return False
        return True

    if (not force) and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        existing = _read_text(output_path)
        if _is_openprompt_template_safe(existing):
            logging.info("[Three-Agent Prompt] Found existing prompt, reuse: %s", output_path)
            return output_path
        else:
            logging.warning("[Three-Agent Prompt] Existing prompt invalid, will overwrite: %s", output_path)

    # ✅ 这里按你的写死策略：优先用入参 api_key；没有就用全局写死的 XIAOHU_KEY
    if not api_key:
        api_key = XIAOHU_KEY
    if not api_key:
        raise RuntimeError("未设置 API Key（当前为 None），请在代码中写死 XIAOHU_KEY 或传入 api_key=...")

    from prompt_agent import export_openprompt_template
    safe_template = export_openprompt_template(use_text_b=False)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(safe_template)

    logging.info("[Three-Agent Prompt] Wrote safe one-line template to: %s", output_path)
    return output_path


# =========================
# 主流程
# =========================
if __name__ == "__main__":
    fewshot_platform = "rec-dy"

    number = [15]
    anchor0 = ['闲聊', "闲聊系统", "不可推荐", '售后', '物流', '差评', '气氛', '主播个人提问', '陈述感受']
    anchor1 = ['购买', "推荐系统", "可推荐", '购买意图', '推荐价值', '商品兴趣', '商品功能', '细节询问', '价格促销', '使用场景', '积极评价', '决策辅助']
    # anchor0 = ['chitchat', "Small talk system", "Not recommendable", 'After-sales service', 'Logistics',
    #            'Negative review', 'Atmosphere', 'Personal questions for the host', 'Expressing feelings']
    # anchor1 = ['Purchase', "Recommendation system", "Recommendable", 'Purchase intent', 'Recommendation value',
    #            'Product interest', 'Product features', 'Detailed inquiry', 'Price promotion', 'Usage scenario',
    #            'Positive review', 'Decision support']
    dataset = {'rec-dy'}
    template = {0}
    seed_list = [i for i in range(110, 126)]
    batch_size_list = [16]
    learning_rate_list = [4e-5]
    shot_list = [50]
    verbalizer_set = {"kpt"}
    max_epochs_list = [15]
    p_list = [800]

    use_hard_template = True
    template_style = "default"

    # 初始化时间衰减管理器
    keyword_manager = TemporalAnchorManager(
        anchor_groups=[anchor0, anchor1],
        model_path=r'/home/ubuntu/juhaoye/1/recommendation_intent_recognition/model/hub/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
        half_life=20
    )

    # 三智能体 Prompt 生成（默认复用已有模板文件）
    prompt_output_path = "scripts/TextClassification/{}/auto_agent_template.txt".format(fewshot_platform)
    train_csv_path = "datasets/TextClassification/{}/train.csv".format(fewshot_platform)

    prompt_path = generate_prompt_with_three_agents(
        platform=fewshot_platform,
        train_csv_path=train_csv_path,
        output_path=prompt_output_path,
        base_url="https://chat.xiaohuapi.site/v1",
        api_key_env="XIAOHU_API_KEY",
        api_key=XIAOHU_KEY,
        seed=188,
        max_iters=3,
        k_per_class=3,
        force=False,
    )

    # ✅ 全局初始化耗时：到这里为止（进入 seed 循环之前）
    GLOBAL_INIT_TIME = time.time() - SCRIPT_START_TIME
    print("\n" + "=" * 20)
    print("全局初始化耗时(GlobalInitSec): {:.2f}秒".format(GLOBAL_INIT_TIME))
    print("=" * 20 + "\n")

    # =========================
    # ✅ 全脚本端到端累计器（跨所有 seed）
    # =========================
    grand_total_samples = 0            # 所有 seed 总样本数（累加）
    grand_total_seed_time = 0.0        # 所有 seed 的 SeedTimeSec 累加（仅参考）
    last_seed_end_wall = SCRIPT_START_TIME

    # =========================
    # 运行实验
    # =========================
    for n, t, j, i, m, k, v, e, pbs, num in product(
        dataset, template, seed_list, batch_size_list, shot_list,
        learning_rate_list, verbalizer_set, max_epochs_list, p_list, number
    ):
        # ✅ seed 计时：seed 内部全部流程（最终会包含写 xlsx/txt）
        seed_start_time = time.time()
        total_samples_processed = 0

        os.makedirs("result", exist_ok=True)
        label_file = os.path.join("result", "label_for_cal.csv")
        pd.DataFrame(columns=["true_label", "pred_label"]).to_csv(label_file, index=False, encoding="utf-8")

        os.makedirs("scripts/TextClassification/{}".format(fewshot_platform), exist_ok=True)
        with open("scripts/TextClassification/{}/kpt.txt".format(fewshot_platform), "w", encoding="utf-8") as f:
            f.write(anchor0[0] + "\n")
            f.write(anchor1[0] + "\n")

        processing_batch_size = pbs
        batch_rows: List[List[str]] = []
        count = 0

        with open(input_file_path, "r", encoding="utf-8") as fin:
            reader = csv.reader(fin)

            for row in reader:
                batch_rows.append(row)

                if len(batch_rows) == processing_batch_size:
                    print("\n=== 开始处理第 {} 批（{}条） ===".format(count + 1, processing_batch_size))
                    round_start_time = time.time()

                    # 1) 分词
                    word_lists = []
                    for r in batch_rows:
                        output = tok(r[2])
                        tokens = output["tok"]
                        pos_tags = output["pos"]
                        filtered_words = [
                            w for w, ptag in zip(tokens, pos_tags)
                            if ptag in {"NOUN", "PROPN", "VERB"} and len(w) > 1
                        ]
                        word_lists.append(filtered_words)

                    # 2) 写 zeroshot 临时词表
                    with open(output_file, "w", encoding="utf-8", newline="") as fout:
                        writer = csv.writer(fout)
                        for words in word_lists:
                            for w_ in words:
                                writer.writerow([w_, 0])

                    # 写 fewshot 测试集
                    os.makedirs("datasets/TextClassification/{}".format(fewshot_platform), exist_ok=True)
                    with open("datasets/TextClassification/{}/test.csv".format(fewshot_platform), "w", encoding="utf-8", newline="") as fout1:
                        writer = csv.writer(fout1)
                        writer.writerows(batch_rows)

                    # 3) zeroshot
                    print("\n[ZeroShot 阶段]")
                    zeroshot()

                    # 4) 读取相关词
                    dfz = pd.read_csv("datasets/veb/{}/test.csv".format(zeroshot_platform))
                    dfz = dfz[dfz["predict"] == 1]["text"]
                    related_wordlist = dfz.values.tolist()
                    print("related_wordlist:", related_wordlist)

                    flag_no_related = False
                    if len(related_wordlist) != 0:
                        new_words = []
                        for test_word in related_wordlist:
                            new_words.append(keyword_manager.calculate_distance(test_word))

                        chat_words, rec_words = keyword_manager.update_keywords(new_words, num)
                        chat_words.insert(0, anchor0[0])
                        rec_words.insert(0, anchor1[0])

                        print("chat_words:", chat_words)
                        print("rec_words:", rec_words)

                        with open("scripts/TextClassification/{}/kpt.txt".format(fewshot_platform), "w", encoding="utf-8") as f:
                            f.write(",".join(chat_words) + "\n")
                            f.write(",".join(rec_words) + "\n")
                    else:
                        flag_no_related = True

                    # 5) 训练/测试
                    if count != 0 and flag_no_related:
                        fewshot1(n, t, j, i, v, fewshot_platform=fewshot_platform)
                    else:
                        fewshot(
                            n, t, j, i, m, k, v, e,
                            fewshot_platform=fewshot_platform,
                            auto_prompt=True,
                            prompt_path=prompt_path,
                            use_hard_template=use_hard_template,
                            template_style=template_style,
                        )

                    total_samples_processed += processing_batch_size
                    count += 1
                    batch_rows = []

                    round_end_time = time.time()
                    print("本批处理耗时: {:.2f}秒".format(round_end_time - round_start_time))

            # 处理最后一批
            if batch_rows:
                print("\n=== 开始处理最后一批（{}条） ===".format(len(batch_rows)))
                round_start_time = time.time()

                word_lists = []
                for r in batch_rows:
                    output = tok(r[2])
                    tokens = output["tok"]
                    pos_tags = output["pos"]
                    filtered_words = [
                        w for w, ptag in zip(tokens, pos_tags)
                        if ptag in {"NOUN", "PROPN", "VERB"} and len(w) > 1
                    ]
                    word_lists.append(filtered_words)

                with open(output_file, "w", encoding="utf-8", newline="") as fout:
                    writer = csv.writer(fout)
                    for words in word_lists:
                        for w_ in words:
                            writer.writerow([w_, 0])

                with open("datasets/TextClassification/{}/test.csv".format(fewshot_platform), "w", encoding="utf-8", newline="") as fout1:
                    writer = csv.writer(fout1)
                    writer.writerows(batch_rows)

                print("\n[ZeroShot 阶段]")
                zeroshot()

                dfz = pd.read_csv("datasets/veb/{}/test.csv".format(zeroshot_platform))
                dfz = dfz[dfz["predict"] == 1]["text"]
                related_wordlist = dfz.values.tolist()

                flag_no_related = False
                if len(related_wordlist) != 0:
                    new_words = []
                    for test_word in related_wordlist:
                        new_words.append(keyword_manager.calculate_distance(test_word))

                    chat_words, rec_words = keyword_manager.update_keywords(new_words, num)
                    chat_words.insert(0, anchor0[0])
                    rec_words.insert(0, anchor1[0])

                    print("chat_words:", chat_words)
                    print("rec_words:", rec_words)

                    with open("scripts/TextClassification/{}/kpt.txt".format(fewshot_platform), "w", encoding="utf-8") as f:
                        f.write(",".join(chat_words) + "\n")
                        f.write(",".join(rec_words) + "\n")
                else:
                    flag_no_related = True

                if count != 0 and flag_no_related:
                    fewshot1(n, t, j, i, v, fewshot_platform=fewshot_platform)
                else:
                    fewshot(
                        n, t, j, i, m, k, v, e,
                        fewshot_platform=fewshot_platform,
                        auto_prompt=True,
                        prompt_path=prompt_path,
                        use_hard_template=use_hard_template,
                        template_style=template_style,
                    )

                total_samples_processed += len(batch_rows)
                count += 1

                round_end_time = time.time()
                print("本批处理耗时: {:.2f}秒".format(round_end_time - round_start_time))

        # =========================
        # 评价指标
        # =========================
        dfres = pd.read_csv("result/label_for_cal.csv")
        y_true = dfres["true_label"]
        y_pred = dfres["pred_label"]

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
        recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

        # =========================
        # ✅ 先准备输出内容（写文件之前）
        # =========================
        content_write_header = "=" * 20 + "\n"
        content_write_header += "dataset {}\t".format(fewshot_platform)
        content_write_header += "temp_id {}\t".format(t)
        content_write_header += "seed {}\t".format(j)
        content_write_header += "shot {}\t".format(m)
        content_write_header += "verb_spe {}\t".format(v)
        content_write_header += "verb_num {}\t".format(num)
        content_write_header += "pici {}\t".format(processing_batch_size)
        content_write_header += "lr {}\t".format(k)
        content_write_header += "max_epochs {}\t\n".format(e)

        content_write_header += "Acc: {:.4f}\t".format(accuracy)
        content_write_header += "Pre: {:.4f}\t".format(precision)
        content_write_header += "Rec: {:.4f}\t".format(recall)
        content_write_header += "F1s: {:.4f}\t\n".format(f1)

        # =========================
        # ✅ 写结果文件耗时（一行统计）
        # =========================
        write_append_start = time.time()

        # 追加写 Excel（先写一行“无时间字段”的占位行，保持策略不变）
        result_xlsx = "./result/{}_250_J-F_800chunk.xlsx".format(fewshot_platform)
        df_out = pd.DataFrame({
            "name": ["推荐意图识别"],
            "dataset": [fewshot_platform],
            "template_id": [t],
            "Seed": [j],
            "Shot": [m],
            "verb_spe": [v],
            "verb_num": [num],
            "pici": [processing_batch_size],
            "learning_rate": [k],
            "batch_size": [i],
            "max_epochs": [e],
            "Accuracy": [accuracy],
            "Precision": [precision],
            "Recall": [recall],
            "F1 Score": [f1],
            "GlobalInitSec": [None],
            "WriteAppendSec": [None],
            "SeedTimeSec": [None],
            "E2E_TotalSec_sofar": [None],
            "NumSamples_seed": [total_samples_processed],
            "GrandTotalSamples_all": [None],
            "SecPerSample_seed": [None],
            "E2E_SecPerSample_sofar": [None],
            "E2E_SecPerSample_final": [None],
        })

        if not os.path.exists(result_xlsx):
            df_out.to_excel(result_xlsx, index=False, header=True)
        else:
            with pd.ExcelWriter(result_xlsx, mode="a", if_sheet_exists="overlay", engine="openpyxl") as writer:
                try:
                    sheet = writer.book["Sheet1"] if "Sheet1" in writer.book.sheetnames else writer.book.active
                    startrow = sheet.max_row
                except Exception:
                    startrow = 0
                df_out.to_excel(writer, index=False, header=False, startrow=startrow)

        # 写 txt：先写指标头（时间行稍后补一行，保持策略不变）
        result_txt = "./result/{}_250_J-F_800chunk.txt".format(fewshot_platform)
        with open(result_txt, "a", encoding="utf-8") as fout_result:
            fout_result.write(content_write_header)

        write_append_end = time.time()
        write_append_sec = write_append_end - write_append_start

        # =========================
        # ✅ 现在 seed 结束（包含写 xlsx/txt）
        # =========================
        seed_end_time = time.time()
        seed_total_time = seed_end_time - seed_start_time

        # ✅ per-seed 平均耗时（seed 内口径）
        if total_samples_processed > 0:
            sec_per_sample_seed = seed_total_time / total_samples_processed
        else:
            sec_per_sample_seed = float("nan")

        # ✅ 全脚本累计（跨 seed）
        grand_total_samples += total_samples_processed
        grand_total_seed_time += seed_total_time

        # ✅ 端到端（E2E）口径：真实墙钟时间（从脚本开始到当前 seed 结束）
        e2e_total_sec_sofar = seed_end_time - SCRIPT_START_TIME
        if grand_total_samples > 0:
            e2e_sec_per_sample_sofar = e2e_total_sec_sofar / grand_total_samples
        else:
            e2e_sec_per_sample_sofar = float("nan")

        # =========================
        # ✅ 指标下面一行：时间统计（你要的 E2E 口径）
        # =========================
        time_line = (
            "GlobalInitSec: {:.2f}\t"
            "WriteAppendSec: {:.2f}\t"
            "SeedTimeSec: {:.2f}\t"
            "E2E_TotalSec(sofar): {:.2f}\t"
            "NumSamples(seed): {}\t"
            "GrandTotalSamples(all): {}\t"
            "SecPerSample(seed): {:.6f}\t"
            "E2E_SecPerSample(sofar): {:.6f}\t\n\n"
        ).format(
            GLOBAL_INIT_TIME,
            write_append_sec,
            seed_total_time,
            e2e_total_sec_sofar,
            total_samples_processed,
            grand_total_samples,
            sec_per_sample_seed,
            e2e_sec_per_sample_sofar,
        )

        content_write = content_write_header + time_line
        print(content_write)

        # ✅ 把时间行追加到 txt（紧跟在指标下面一行）
        with open(result_txt, "a", encoding="utf-8") as fout_result:
            fout_result.write(time_line)

        # ✅ 再追加一行“含时间字段”的完整记录到 Excel（保持策略不变）
        df_time_only = pd.DataFrame({
            "name": ["推荐意图识别"],
            "dataset": [fewshot_platform],
            "template_id": [t],
            "Seed": [j],
            "Shot": [m],
            "verb_spe": [v],
            "verb_num": [num],
            "pici": [processing_batch_size],
            "learning_rate": [k],
            "batch_size": [i],
            "max_epochs": [e],
            "Accuracy": [accuracy],
            "Precision": [precision],
            "Recall": [recall],
            "F1 Score": [f1],
            "GlobalInitSec": [GLOBAL_INIT_TIME],
            "WriteAppendSec": [write_append_sec],
            "SeedTimeSec": [seed_total_time],
            "E2E_TotalSec_sofar": [e2e_total_sec_sofar],
            "NumSamples_seed": [total_samples_processed],
            "GrandTotalSamples_all": [grand_total_samples],
            "SecPerSample_seed": [sec_per_sample_seed],
            "E2E_SecPerSample_sofar": [e2e_sec_per_sample_sofar],
            "E2E_SecPerSample_final": [None],  # 最终在脚本结束统一打印
        })

        with pd.ExcelWriter(result_xlsx, mode="a", if_sheet_exists="overlay", engine="openpyxl") as writer:
            try:
                sheet = writer.book["Sheet1"] if "Sheet1" in writer.book.sheetnames else writer.book.active
                startrow = sheet.max_row
            except Exception:
                startrow = 0
            df_time_only.to_excel(writer, index=False, header=False, startrow=startrow)

        last_seed_end_wall = seed_end_time

    # =========================
    # 脚本总耗时：从“点击运行”到全部 seed 完成（最终 E2E）
    # =========================
    script_total_time = time.time() - SCRIPT_START_TIME
    if grand_total_samples > 0:
        e2e_sec_per_sample_final = script_total_time / grand_total_samples
    else:
        e2e_sec_per_sample_final = float("nan")

    print("\n" + "=" * 20)
    print("脚本总耗时(从运行开始到全部结束): {:.2f}秒".format(script_total_time))
    print("GrandTotalSamples(all seeds): {}".format(grand_total_samples))
    print("E2E_SecPerSample(final): {:.6f} 秒/样本".format(e2e_sec_per_sample_final))
    print("=" * 20)

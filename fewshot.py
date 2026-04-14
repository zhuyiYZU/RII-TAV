import tqdm
from openprompt.data_utils.text_classification_dataset import *
import torch
from openprompt.data_utils.utils import InputExample
from sklearn.metrics import *
import argparse
import pandas as pd
import numpy as np
import csv

from openprompt import PromptDataLoader
from openprompt.prompts import ManualVerbalizer, KnowledgeableVerbalizer, SoftVerbalizer, AutomaticVerbalizer
from openprompt.prompts import ManualTemplate, PtuningTemplate
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

parser = argparse.ArgumentParser("")
parser.add_argument("--shot", type=int, default=50)
parser.add_argument("--seed", type=int, default=123)
parser.add_argument("--plm_eval_mode", action="store_true")
parser.add_argument("--model", type=str, default='bert') # bert,roberta,t5
parser.add_argument("--model_name_or_path", default='/home/ubuntu/juhaoye/1/recommendation_intent_recognition/models/chinese-roberta-wwm-ext')#bert分词器
parser.add_argument("--verbalizer", type=str)
parser.add_argument("--calibration", action="store_true")
parser.add_argument("--filter", default="none", type=str)
parser.add_argument("--template_id", type=int)
parser.add_argument("--dataset", type=str)
parser.add_argument("--result_file", type=str, default="../sfs_scripts/results_fewshot_manual_kpt.txt")
parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
parser.add_argument("--max_epochs", type=int, default=20)
parser.add_argument("--kptw_lr", default=0.05, type=float)
parser.add_argument("--pred_temp", default=1.0, type=float)
parser.add_argument("--max_token_split", default=-1, type=int)
parser.add_argument("--learning_rate", default=4e-5, type=float)
parser.add_argument("--xiaohu_base_url", type=str, default="https://chat.xiaohuapi.site/v1")
parser.add_argument("--xiaohu_api_key", type=str, default="sk-NDweZpKBL5cOikWhW25dv3bcLuUQtiYGMc4AJ99EEj3bKuch")  # 直接写死 API 密钥
parser.add_argument("--batch_size", default=16, type=int)
parser.add_argument("--auto_prompt", action="store_true", help="Auto-generate prompt from training data (will use ManualTemplate/硬模板 instead of PtuningTemplate/软模板)")
parser.add_argument("--prompt_template_path", type=str, default=None, help="Path to save/load auto-generated prompt")
parser.add_argument("--use_hard_template", action="store_true", help="Use hard template (简洁硬模板) - default True when auto_prompt is enabled")
parser.add_argument("--template_style", type=str, default="default", choices=["default", "customer_service", "analyst", "classifier", "intent"], help="Style for hard template")
parser.add_argument("--use_ace", action="store_true", help="Use ACE framework for Generate-Reflect-Curate cycle")
parser.add_argument("--ace_auto_curate", action="store_true", help="Automatically update playbook based on reflection (requires --use_ace)")
# parser.add_argument("--manual_or_soft", default="manual", type=str)
args = parser.parse_args()

import random

this_run_unicode = "model"

from openprompt.utils.reproduciblity import set_seed

set_seed(args.seed)

from openprompt.plms import load_plm

plm, tokenizer, model_config, WrapperClass = load_plm(args.model, args.model_name_or_path)

dataset = {}




if args.dataset == "rec-dy":
    dataset['train'] = RecdyProcessor().get_train_examples("datasets/TextClassification/rec-dy/")
    dataset['test'] = RecdyProcessor().get_test_examples("datasets/TextClassification/rec-dy/")
    class_labels = RecdyProcessor().get_labels()
    scriptsbase = "TextClassification/rec-dy"
    scriptformat = "txt"
    cutoff=0.5
    max_seq_l = 128
    batch_s = args.batch_size
elif args.dataset == "rec-ks":
    dataset['train'] = RecdyProcessor().get_train_examples("datasets/TextClassification/rec-ks/")
    dataset['test'] = RecdyProcessor().get_test_examples("datasets/TextClassification/rec-ks/")
    class_labels = RecdyProcessor().get_labels()
    scriptsbase = "TextClassification/rec-ks"
    scriptformat = "txt"
    cutoff = 0.5
    max_seq_l = 128
    batch_s = args.batch_size
elif args.dataset == "rec-xhs":
    dataset['train'] = RecdyProcessor().get_train_examples("datasets/TextClassification/rec-xhs/")
    dataset['test'] = RecdyProcessor().get_test_examples("datasets/TextClassification/rec-xhs/")
    class_labels = RecdyProcessor().get_labels()
    scriptsbase = "TextClassification/rec-xhs"
    scriptformat = "txt"
    cutoff = 0.5
    max_seq_l = 128
    batch_s = args.batch_size
elif args.dataset == "rec-new":
    dataset['train'] = RecnewProcessor().get_train_examples("datasets/TextClassification/rec-new/")
    dataset['test'] = RecnewProcessor().get_test_examples("datasets/TextClassification/rec-new/")
    class_labels = RecnewProcessor().get_labels()
    scriptsbase = "TextClassification/rec-new"
    scriptformat = "txt"
    cutoff = 0.5
    max_seq_l = 128
    batch_s = args.batch_size
else:
    raise NotImplementedError
def load_single_line_template(path: str) -> str:
    """读取模板文件：取第一个非空行，并做基础清洗/括号配对检查。"""
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines()]

    # 去掉空行
    lines = [ln for ln in lines if ln]

    if not lines:
        raise ValueError(f"Template file is empty: {path}")

    # 只取第一条模板（OpenPrompt from_file 也可以多条，这里更稳）
    text = lines[0]

    # 去掉 markdown 围栏（防止 LLM 输出 ```json ... ```）
    text = text.replace("```json", "").replace("```", "").strip()

    # 简单括号配对检查（只检查数量；更严格可以用栈）
    if text.count("{") != text.count("}"):
        raise ValueError(
            "Template has unmatched '{' '}' braces.\n"
            f"count('{{'))={text.count('{')}, count('}}')={text.count('}')}\n"
            f"template preview: {text[:200]}"
        )

    return text

if args.auto_prompt:
    import os
    from prompt_agent import PromptOptimizationAgent, BASE_PLAYBOOK
    from llm_utils import XiaoHuAPIClient

    # 生成优化后的硬模板并保存
    if args.prompt_template_path:
        prompt_template_path = args.prompt_template_path
    else:
        prompt_template_path = f"./scripts/{scriptsbase}/auto_agent_template.txt"

    # ✅ 关键修复：如果 prompt 文件已经存在，就直接复用，不再请求 LLM
    if os.path.exists(prompt_template_path) and os.path.getsize(prompt_template_path) > 20:
        print("\n" + "=" * 50)
        print(f">>> 检测到已有 Prompt，直接复用，不再请求 LLM: {prompt_template_path}")
        print("=" * 50 + "\n")

        from openprompt.prompts import ManualTemplate

        template_text = load_single_line_template(args.prompt_template_path)
        mytemplate = ManualTemplate(tokenizer=tokenizer, text=template_text)

        print(f"Loaded optimized hard template from: {prompt_template_path}")

    else:
        # 走三智能体生成（需要联网）
        api_key = args.xiaohu_api_key
        if not api_key:
            raise RuntimeError(
                "API 密钥未找到：\n"
                "1) 推荐：export XIAOHU_API_KEY='sk-你的key'\n"
                "2) 或者：在命令行传 --xiaohu_api_key\n"
            )

        # 创建客户端并初始化智能体
        client = XiaoHuAPIClient(
            api_key=api_key,
            base_url=args.xiaohu_base_url,  # ✅ 这里必须是 https://chat.xiaohuapi.site/v1
            model="gpt-4o",
            timeout=30,
        )
        agent = PromptOptimizationAgent(
            client=client,
            base_playbook=BASE_PLAYBOOK,
            text_a_col=2,
            label_col=3,
            text_b_col=None,  # 你的 train.csv 若有解释列再改成对应 index
        )

        print("\n" + "=" * 50)
        print(">>> 启动三智能体进化流程：Generator -> Reflector -> Integrator")
        print(">>> 基于训练集抽样（3正3负）进行多轮反思整合")
        print("=" * 50)

        train_csv_path = f"datasets/TextClassification/{args.dataset}/train.csv"

        optimized_playbook = agent.run_optimization(
            train_csv_path=train_csv_path,
            max_iters=3,
            seed=args.seed,
            k_per_class=3,
        )

        os.makedirs(os.path.dirname(prompt_template_path), exist_ok=True)
        with open(prompt_template_path, "w", encoding="utf-8") as f:
            f.write(optimized_playbook)

        print(f">>> 三智能体自动生成/进化后的 Prompt 已保存至: {prompt_template_path}")
        print("=" * 50 + "\n")

        from openprompt.prompts import ManualTemplate

        template_text = load_single_line_template(args.prompt_template_path)
        mytemplate = ManualTemplate(tokenizer=tokenizer, text=template_text)

        print(f"Loaded optimized hard template from: {prompt_template_path}")

else:
    # 原始的软模板加载逻辑 (PtuningTemplate)
    mytemplate = PtuningTemplate(model=plm, tokenizer=tokenizer).from_file(
        f"./scripts/{scriptsbase}/ptuning_template.txt",
        choice=args.template_id
    )
    print(f"Using original soft template (PtuningTemplate), template_id={args.template_id}")

if args.verbalizer == "kpt":
    myverbalizer = KnowledgeableVerbalizer(tokenizer, classes=class_labels, candidate_frac=cutoff,
                                           pred_temp=args.pred_temp, max_token_split=args.max_token_split).from_file(
        path=f"./scripts/{scriptsbase}/kpt.{scriptformat}")
elif args.verbalizer == "manual":
    myverbalizer = ManualVerbalizer(tokenizer, classes=class_labels).from_file(
        f"./scripts/{scriptsbase}/manual_verbalizer.{scriptformat}")
elif args.verbalizer == "kpt++":
    myverbalizer = ManualVerbalizer(tokenizer, classes=class_labels).from_file(
        f"./scripts/{scriptsbase}/kpt++.{scriptformat}")


# (contextual) calibration
if args.verbalizer in ["kpt", "manual","kpt++"]:
    if args.calibration or args.filter != "none":
        from openprompt.data_utils.data_sampler import FewShotSampler

        support_sampler = FewShotSampler(num_examples_total=200, also_sample_dev=False)
        dataset['support'] = support_sampler(dataset['train'], seed=args.seed)

        # for example in dataset['support']:
        #     example.label = -1 # remove the labels of support set for clarification
        support_dataloader = PromptDataLoader(dataset=dataset["support"], template=mytemplate, tokenizer=tokenizer,
                                              tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_l,
                                              decoder_max_length=3,
                                              batch_size=batch_s, shuffle=False, teacher_forcing=False,
                                              predict_eos_token=False,
                                              truncate_method="tail")

from openprompt import PromptForClassification

use_cuda = True
prompt_model = PromptForClassification(plm=plm, template=mytemplate, verbalizer=myverbalizer, freeze_plm=False,
                                       plm_eval_mode=args.plm_eval_mode)
if use_cuda:
    prompt_model = prompt_model.cuda()

# HP
# if args.calibration:
if args.verbalizer in ["kpt","manual","kpt++"]:
    if args.calibration or args.filter != "none":
        org_label_words_num = [len(prompt_model.verbalizer.label_words[i]) for i in range(len(class_labels))]
        from openprompt.utils.calibrate import calibrate

        # calculate the calibration logits
        cc_logits = calibrate(prompt_model, support_dataloader)
        print("the calibration logits is", cc_logits)
        print("origial label words num {}".format(org_label_words_num))

    if args.calibration:
        myverbalizer.register_calibrate_logits(cc_logits)
        new_label_words_num = [len(myverbalizer.label_words[i]) for i in range(len(class_labels))]
        print("After filtering, number of label words per class: {}".format(new_label_words_num))

    # register the logits to the verbalizer so that the verbalizer will divide the calibration probability in producing label logits
  # currently, only ManualVerbalizer and KnowledgeableVerbalizer support calibration.

#### sannhang henzhongyao！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
################！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
from openprompt.data_utils.data_sampler import FewShotSampler
sampler = FewShotSampler(num_examples_per_label=args.shot, also_sample_dev=True, num_examples_per_label_dev=args.shot)
dataset['train'], dataset['validation'] = sampler(dataset['train'], seed=args.seed)
# trainData = dataset['train']
# dataset['train'], dataset['validation'] = sampler(trainData, seed=args.seed)
# dataset['train1'], dataset['validation'] = sampler(trainData, seed=random.randint(1,100))
######################################################################################！！！！！！！！！！！！！！！！！！！！！！！！！！！

train_dataloader = PromptDataLoader(dataset=dataset["train"], template=mytemplate, tokenizer=tokenizer,
                                    tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_l,
                                    decoder_max_length=3,
                                    batch_size=batch_s, shuffle=True, teacher_forcing=False, predict_eos_token=False,
                                    truncate_method="tail")

validation_dataloader = PromptDataLoader(dataset=dataset["validation"], template=mytemplate, tokenizer=tokenizer,
                                         tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_l,
                                         decoder_max_length=3,
                                         batch_size=batch_s, shuffle=False, teacher_forcing=False,
                                         predict_eos_token=False,
                                         truncate_method="tail")

# zero-shot test
test_dataloader = PromptDataLoader(dataset=dataset["test"], template=mytemplate, tokenizer=tokenizer,
                                   tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_l, decoder_max_length=3,
                                   batch_size=batch_s, shuffle=False, teacher_forcing=False, predict_eos_token=False,
                                   truncate_method="tail")
'''
import numpy as np

# 计算混淆矩阵
def compute_confusion_matrix(precited,expected):
    part = precited ^ expected             # 对结果进行分类，亦或使得判断正确的为0,判断错误的为1
    pcount = np.bincount(part)             # 分类结果统计，pcount[0]为0的个数，pcount[1]为1的个数
    tp_list = list(precited & expected)    # 将TP的计算结果转换为list
    fp_list = list(precited & ~expected)   # 将FP的计算结果转换为list
    tp = tp_list.count(1)                  # 统计TP的个数
    fp = fp_list.count(1)                  # 统计FP的个数
    tn = pcount[0] - tp                    # 统计TN的个数
    fn = pcount[1] - fp                    # 统计FN的个数
    return tp, fp, tn, fn

# 计算常用指标
def compute_indexes(tp, fp, tn, fn):
    if (tp+tn+fp+fn)!=0 & (tp+fp)!=0 & (tp+fn)!= 0:
        accuracy = (tp+tn) / (tp+tn+fp+fn)     # 准确率
        precision = tp / (tp+fp)               # 精确率
        recall = tp / (tp+fn)                  # 召回率
        F1 = (2*precision*recall) / (precision+recall)    # F1
    else:
        accuracy = 0.1     # 准确率
        precision = 0.1               # 精确率
        recall = 0.1                  # 召回率
        F1 = 0.1    # F1
    return accuracy, precision, recall, F1
'''
def evaluate(prompt_model, dataloader, desc):
    prompt_model.eval()
    allpreds = []
    alllabels = []
    pbar = tqdm.tqdm(dataloader, desc=desc)
    for step, inputs in enumerate(pbar):

        inputs = inputs.cuda()
        logits = prompt_model(inputs)
        labels = inputs['label']
        alllabels.extend(labels.cpu().tolist())
        allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())

    acc = accuracy_score(alllabels,allpreds)
    return acc
def evaluate1(prompt_model, dataloader, desc):
    prompt_model.eval()
    allpreds = []
    alllabels = []
    allltexts = []
    pbar = tqdm.tqdm(dataloader, desc=desc)
    for step, inputs in enumerate(pbar):

        inputs = inputs.cuda()
        logits = prompt_model(inputs)

        # texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in inputs['title']]
        labels = inputs['label']
        alllabels.extend(labels.cpu().tolist())
        allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
    acc = accuracy_score(alllabels, allpreds)
    pre = precision_score(alllabels,allpreds,average='macro')
    recall = recall_score(alllabels,allpreds,average='macro')
    f1 = f1_score(alllabels,allpreds,average='macro')
    # print(alllabels)
    # print(allpreds[0])
    # 写入 CSV 文件（每行一个值）

    with open(f'result/label_for_cal.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        for i in range(len(allpreds)):
            writer.writerow([alllabels[i],allpreds[i]])  # 每个值单独一行


    print("预测结果已写入 label_for_cal.csv")
    return [acc,pre,recall,f1]
    # # 3. 读取原始测试数据（无列名）
    # df_test = pd.read_csv("datasets/TextClassification/rec-dy/test.csv", header=None)
    #
    # # 3. 验证数据匹配
    # if len(df_test) != len(allpreds):
    #     raise ValueError(f"数据量不匹配: 测试集{len(df_test)}行, 预测结果{len(allpreds)}个")
    #
    # # 4. 将预测结果替换第7列（索引6）
    #
    # df_test.iloc[:,7] = allpreds
    #
    # # 5. 合并到训练集（不去重）
    # train_path = "datasets/TextClassification/rec-dy/train.csv"
    # if os.path.exists(train_path):
    #     # 读取现有训练集
    #     df_train = pd.read_csv(train_path, header=None)
    #     # 直接追加新数据（不去重）
    #     df_merged = pd.concat([df_train, df_test], ignore_index=True)
    # else:
    #     # 如果训练集不存在，直接使用测试集作为新训练集
    #     df_merged = df_test
    # print(df_train[:3].head(5))
    # # 6. 保存（不写入列名）
    # df_merged.to_csv(train_path, index=False, header=False)
    # print(f"已合并数据: 新增{len(df_test)}行 | 训练集总行数: {len(df_merged)}")
    # print(f"第7列已替换为预测结果，示例:")

    # return

############
#############
###############

from transformers import  get_linear_schedule_with_warmup
from torch.optim import AdamW
# class_weights =torch.tensor([0.1,0.9])
# if use_cuda:
#     class_weights = class_weights.cuda()
loss_func = torch.nn.CrossEntropyLoss()


def prompt_initialize(verbalizer, prompt_model, init_dataloader):
    dataloader = init_dataloader
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Init_using_{}".format("train")):
            batch = batch.cuda()
            logits = prompt_model(batch)
        verbalizer.optimize_to_initialize()




if args.verbalizer == "kpt":
    no_decay = ['bias', 'LayerNorm.weight']

    # it's always good practice to set no decay to biase and LayerNorm parameters
    optimizer_grouped_parameters1 = [
        {'params': [p for n, p in prompt_model.plm.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in prompt_model.plm.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]

    # Using different optimizer for prompt parameters and model parameters

    # optimizer_grouped_parameters2 = [
    #     {'params': , "lr":1e-1},
    # ]
    optimizer1 = AdamW(optimizer_grouped_parameters1, lr=float(args.learning_rate))
    optimizer2 = AdamW(prompt_model.verbalizer.parameters(), lr=args.kptw_lr)
    # print(optimizer_grouped_parameters2)

    tot_step = len(train_dataloader) // args.gradient_accumulation_steps * args.max_epochs
    scheduler1 = get_linear_schedule_with_warmup(
        optimizer1,
        num_warmup_steps=0, num_training_steps=tot_step)

    # scheduler2 = get_linear_schedule_with_warmup(
    #     optimizer2,
    #     num_warmup_steps=0, num_training_steps=tot_step)
    scheduler2 = None


elif args.verbalizer == "manual":
    no_decay = ['bias', 'LayerNorm.weight']

    # it's always good practice to set no decay to biase and LayerNorm parameters
    optimizer_grouped_parameters1 = [
        {'params': [p for n, p in prompt_model.plm.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in prompt_model.plm.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]

    # Using different optimizer for prompt parameters and model parameters

    optimizer1 = AdamW(optimizer_grouped_parameters1, lr=float(args.learning_rate))
    print("lr:",optimizer1.param_groups[0]['lr'])
    tot_step = len(train_dataloader) // args.gradient_accumulation_steps * args.max_epochs
    scheduler1 = get_linear_schedule_with_warmup(
        optimizer1,
        num_warmup_steps=0, num_training_steps=tot_step)

    optimizer2 = None
    scheduler2 = None


elif args.verbalizer == "ex_re1":
    no_decay = ['bias', 'LayerNorm.weight']

    # it's always good practice to set no decay to biase and LayerNorm parameters
    optimizer_grouped_parameters1 = [
        {'params': [p for n, p in prompt_model.plm.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in prompt_model.plm.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]

    # Using different optimizer for prompt parameters and model parameters

    optimizer1 = AdamW(optimizer_grouped_parameters1, lr=float(args.learning_rate))

    tot_step = len(train_dataloader) // args.gradient_accumulation_steps * args.max_epochs
    scheduler1 = get_linear_schedule_with_warmup(
        optimizer1,
        num_warmup_steps=0, num_training_steps=tot_step)

    optimizer2 = None
    scheduler2 = None



tot_loss = 0
log_loss = 0
best_val_acc = 0
for epoch in range(args.max_epochs):
    tot_loss = 0
    prompt_model.train()
    for step, inputs in enumerate(train_dataloader):
        if use_cuda:
            inputs = inputs.cuda()
        logits = prompt_model(inputs)
        labels = inputs['label']
        loss = loss_func(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(prompt_model.parameters(), 1.0)
        tot_loss = tot_loss + loss.item()
        optimizer1.step()
        scheduler1.step()
        optimizer1.zero_grad()
        if optimizer2 is not None:
            optimizer2.step()
            optimizer2.zero_grad()
        if scheduler2 is not None:
            scheduler2.step()

    cal_data = evaluate(prompt_model, validation_dataloader, desc="Valid")
    val_acc = cal_data
    if val_acc >= best_val_acc:
        torch.save(prompt_model.state_dict(), f"./ckpts/{this_run_unicode}.ckpt")
        best_val_acc = val_acc
    print("Epoch {}, val_acc {}, tot_loss {}".format(epoch, val_acc, tot_loss), flush=True)

prompt_model.load_state_dict(torch.load(f"./ckpts/{this_run_unicode}.ckpt"))
prompt_model = prompt_model.cuda()
data_set = evaluate1(prompt_model, test_dataloader, desc="Test")


# test_acc = data_set[0]
# test_pre = data_set[1]
# test_recall = data_set[2]
# test_F1scall = data_set[3]
#
# content_write = "=" * 20 + "\n"
# content_write += f"dataset {args.dataset}\t"
# content_write += f"temp {args.template_id}\t"
# content_write += f"seed {args.seed}\t"
# content_write += f"shot {args.shot}\t"
# content_write += f"verb {args.verbalizer}\t"
# content_write += f"cali {args.calibration}\t"
# content_write += f"filt {args.filter}\t"
# content_write += f"maxsplit {args.max_token_split}\t"
# content_write += f"kptw_lr {args.kptw_lr}\t"
# content_write += f"bt {args.batch_size}\t"
# content_write += f"lr {args.learning_rate}\t"
# content_write += f"epoch {args.max_epochs}\t"
#
# content_write += "\n"
#
# content_write += f"Acc: {test_acc}\t"
# content_write += f"Pre: {test_pre}\t"
# content_write += f"Rec: {test_recall}\t"
# content_write += f"F1s: {test_F1scall}\t"
# content_write += "\n\n"
#
# print(content_write)
#
# # with open("./result.txt","w",encoding="utf-8") as file:
# #     file.write(content_write)
#
#
# name = '对比'
# data = {
#     'name': [name],
#     'veb': [args.verbalizer],
#     'dataset': [args.dataset],
#     'Seed': [args.seed],
#     'Shot': [args.shot],
#     'learning_rate': [args.learning_rate],
#     'batch_size': [args.batch_size],
#     'max_epochs': [args.max_epochs],
#     'Accuracy': [test_acc],
#     'Precision': [test_pre],
#     'Recall': [test_recall],
#     'F1 Score': [test_F1scall]
# }
# df = pd.DataFrame(data)
# # df.to_excel(f'./result/{args.dataset}_metrics_data.xlsx')
#
# file_path = f'./result/{args.dataset}_metrics_data.xlsx'
# if not os.path.exists(file_path):
#     df.to_excel(file_path, index=False,header=True)
# else:
#     with pd.ExcelWriter(file_path, mode='a', if_sheet_exists='overlay') as writer:
#         df.to_excel(writer, index=False, header=False, startrow=writer.sheets['Sheet1'].max_row)
#
# with open(f"{args.result_file}", "a") as fout:
#     fout.write(content_write)

import os

# os.remove(f"./ckpts/{this_run_unicode}.ckpt")
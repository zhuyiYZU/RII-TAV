import tqdm
from openprompt.data_utils.text_classification_dataset import *
import torch
from openprompt.data_utils.utils import InputExample
from sklearn.metrics import *
import argparse
import pandas as pd
import numpy as np

from openprompt import PromptDataLoader
from openprompt.prompts import ManualVerbalizer, KnowledgeableVerbalizer, SoftVerbalizer, AutomaticVerbalizer
from openprompt.prompts import ManualTemplate, PtuningTemplate

parser = argparse.ArgumentParser("")
parser.add_argument("--shot", type=int, default=5)
parser.add_argument("--seed", type=int, default=144)
parser.add_argument("--plm_eval_mode", action="store_true")
parser.add_argument("--model", type=str, default='bert') # bert,roberta,t5
parser.add_argument("--model_name_or_path", default='/home/ubuntu/juhaoye/1/recommendation_intent_recognition/models/chinese-roberta-wwm-ext')#bert分词器
parser.add_argument("--verbalizer", type=str)
parser.add_argument("--calibration", action="store_true")
parser.add_argument("--filter", default="none", type=str)
parser.add_argument("--template_id", type=int)
parser.add_argument("--dataset", type=str)
parser.add_argument("--result_file", type=str, default="./label/label.txt")
parser.add_argument("--label_result", type=str, default="datasets/TextClassification/DuDialRec/dev.csv")
parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
parser.add_argument("--max_epochs", type=int, default=15)
parser.add_argument("--kptw_lr", default=0.04, type=float)
parser.add_argument("--pred_temp", default=1.0, type=float)
parser.add_argument("--max_token_split", default=-1, type=int)
parser.add_argument("--learning_rate", default=4e-5, type=str)
parser.add_argument("--batch_size", default=16, type=int)

args = parser.parse_args()

import random

this_run_unicode = str(random.randint(0, 1e10))

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


# mytemplate = ManualTemplate(tokenizer=tokenizer).from_file(path=f"./scripts/{scriptsbase}/manual_template.txt",choice=args.template_id)
mytemplate = PtuningTemplate(model=plm, tokenizer=tokenizer).from_file(f"./scripts/{scriptsbase}/ptuning_template.txt", choice=args.template_id)

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
elif args.verbalizer == "ex_re1":#一个中心点的阔删
    myverbalizer = KnowledgeableVerbalizer(tokenizer, classes=class_labels,candidate_frac=cutoff,
                                           pred_temp=args.pred_temp, max_token_split=args.max_token_split).from_file(
        path=f"./scripts/{scriptsbase}/expand_refine1_verbalizer.{scriptformat}")


# (contextual) calibration
if args.verbalizer in ["kpt","kpt1","kpt2","kpt3","kpt4","kpt5", "manual",'ex_re1','kpt++']:
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
if args.verbalizer in ["kpt", "kpt1","kpt2","kpt3","kpt4","kpt5","manual",'ex_re1','kpt++']:
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
        if use_cuda:
            inputs = inputs.cuda()
        logits = prompt_model(inputs)
        labels = inputs['label']
        alllabels.extend(labels.cpu().tolist())
        allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())

    pd.DataFrame({"alllabels":alllabels,"allpreds":allpreds}).to_csv("result.csv", header=0,index=False)
    # print(alllabels)
    # print(allpreds)

    # allpreds = np.array(allpreds)
    # alllabels = np.array(alllabels)
    # tp, fp, tn, fn = compute_confusion_matrix(allpreds, alllabels)
    #
    # acc, pre, recall, F1score = compute_indexes(tp, fp, tn, fn)



    # acc = sum([int(i != j) for i, j in zip(allpreds, alllabels)]) / len(allpreds)
    acc = accuracy_score(alllabels, allpreds)
    pre = precision_score(alllabels, allpreds, average='macro')
    recall = recall_score(alllabels, allpreds, average='macro')
    F1score = f1_score(alllabels, allpreds, average='macro')
    # pre, recall, F1score, _ = precision_recall_fscore_support(alllabels, allpreds, average='macro')
    cal_data = [acc, pre, recall, F1score]
    return cal_data


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




if args.verbalizer == "kpt"or"kpt1"or"kpt2"or"kpt3"or"kpt4"or"kpt5":
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
    val_acc = cal_data[0]
    if val_acc >= best_val_acc:
        torch.save(prompt_model.state_dict(), f"./ckpts/{this_run_unicode}.ckpt")
        best_val_acc = val_acc
    print("Epoch {}, val_acc {}, tot_loss {}".format(epoch, val_acc, tot_loss), flush=True)

prompt_model.load_state_dict(torch.load(f"./ckpts/{this_run_unicode}.ckpt"))
prompt_model = prompt_model.cuda()
data_set = evaluate(prompt_model, test_dataloader, desc="Test")
test_acc = data_set[0]
test_pre = data_set[1]
test_recall = data_set[2]
test_F1scall = data_set[3]

content_write = "=" * 20 + "\n"
content_write += f"dataset {args.dataset}\t"
content_write += f"temp {args.template_id}\t"
content_write += f"seed {args.seed}\t"
content_write += f"shot {args.shot}\t"
content_write += f"verb {args.verbalizer}\t"
content_write += f"cali {args.calibration}\t"
content_write += f"filt {args.filter}\t"
content_write += f"maxsplit {args.max_token_split}\t"
content_write += f"kptw_lr {args.kptw_lr}\t"
content_write += f"bt {args.batch_size}\t"
content_write += f"lr {args.learning_rate}\t"
content_write += f"epoch {args.max_epochs}\t"

content_write += "\n"

content_write += f"Acc: {test_acc}\t"
content_write += f"Pre: {test_pre}\t"
content_write += f"Rec: {test_recall}\t"
content_write += f"F1s: {test_F1scall}\t"
content_write += "\n\n"

print(content_write)

# with open("./result.txt","w",encoding="utf-8") as file:
#     file.write(content_write)


name = '对比'
data = {
    'name': [name],
    'veb': [args.verbalizer],
    'dataset': [args.dataset],
    'Seed': [args.seed],
    'Shot': [args.shot],
    'learning_rate': [args.learning_rate],
    'batch_size': [args.batch_size],
    'max_epochs': [args.max_epochs],
    'Accuracy': [test_acc],
    'Precision': [test_pre],
    'Recall': [test_recall],
    'F1 Score': [test_F1scall]
}
df = pd.DataFrame(data)
# df.to_excel(f'./result/{args.dataset}_metrics_data.xlsx')

file_path = f'./result/{args.dataset}_metrics_data.xlsx'
if not os.path.exists(file_path):
    df.to_excel(file_path, index=False,header=True)
else:
    with pd.ExcelWriter(file_path, mode='a', if_sheet_exists='overlay') as writer:
        df.to_excel(writer, index=False, header=False, startrow=writer.sheets['Sheet1'].max_row)

with open(f"{args.result_file}", "a") as fout:
    fout.write(content_write)

import os

os.remove(f"./ckpts/{this_run_unicode}.ckpt")
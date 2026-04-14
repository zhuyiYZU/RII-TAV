import os
import tqdm
from openprompt.data_utils.text_classification_dataset import RecdyProcessor,RecnewProcessor
import torch
import pandas as pd
import argparse
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

from openprompt import PromptDataLoader
from openprompt.prompts import ManualVerbalizer, KnowledgeableVerbalizer
from openprompt.prompts import ManualTemplate, PtuningTemplate

parser = argparse.ArgumentParser("")
parser.add_argument("--shot", type=int, default=5)
parser.add_argument("--seed", type=int, default=144)

parser.add_argument("--plm_eval_mode", action="store_true")
parser.add_argument("--model", type=str, default='bert')
parser.add_argument("--model_name_or_path", default='/home/ubuntu/juhaoye/beifen/recommendation_intent_recognition/models/bert_chinese')
# parser.add_argument("--model_name_or_path", default='/home/zy-4090-1/hqq/recommendation_intent_recognition/model/chinese-roberta-wwm-ext')
# parser.add_argument("--model_name_or_path", default='/home/zy-4090-1/models/bert_base_cased')
parser.add_argument("--verbalizer", type=str)
parser.add_argument("--calibration", action="store_true")
parser.add_argument("--filter", default="none", type=str)
parser.add_argument("--template_id", type=int)
parser.add_argument("--dataset", type=str)
parser.add_argument("--result_file", type=str, default="../sfs_scripts/results_fewshot_manual_kpt.txt")
parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
parser.add_argument("--max_epochs", type=int, default=15)
parser.add_argument("--kptw_lr", default=0.05, type=float)
parser.add_argument("--pred_temp", default=1.0, type=float)
parser.add_argument("--max_token_split", default=-1, type=int)
parser.add_argument("--learning_rate", default=2e-5, type=str)
parser.add_argument("--batch_size", default=16, type=int)
# parser.add_argument("--manual_or_soft", default="manual", type=str)
args = parser.parse_args()



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


#mytemplate = ManualTemplate(tokenizer=tokenizer).from_file(path=f"./scripts/{scriptsbase}/manual_template.txt",choice=args.template_id)
mytemplate = PtuningTemplate(model=plm, tokenizer=tokenizer).from_file(f"./scripts/{scriptsbase}/ptuning_template.txt", choice=args.template_id)


if args.verbalizer == "kpt":
    myverbalizer = KnowledgeableVerbalizer(tokenizer, classes=class_labels, candidate_frac=cutoff,
                                           pred_temp=args.pred_temp, max_token_split=args.max_token_split).from_file(
        path=f"./scripts/{scriptsbase}/kpt.{scriptformat}")
elif args.verbalizer == "manual":
    myverbalizer = ManualVerbalizer(tokenizer, classes=class_labels).from_file(
        f"./scripts/{scriptsbase}/manual_verbalizer.{scriptformat}")


# (contextual) calibration
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
                                          truncate_method="tail",num_workers=16,  # 根据CPU核心数调整
    pin_memory=True)




from openprompt import PromptForClassification


prompt_model = PromptForClassification(plm=plm, template=mytemplate, verbalizer=myverbalizer, freeze_plm=False,
                                       plm_eval_mode=args.plm_eval_mode)

prompt_model = prompt_model.cuda()



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


from openprompt.data_utils.data_sampler import FewShotSampler
sampler = FewShotSampler(num_examples_per_label=args.shot, also_sample_dev=True, num_examples_per_label_dev=args.shot)
dataset['train'], dataset['validation'] = sampler(dataset['train'], seed=args.seed)
######################################################################################！！！！！！！！！！！！！！！！！！！！！！！！！！！

train_dataloader = PromptDataLoader(dataset=dataset["train"], template=mytemplate, tokenizer=tokenizer,
                                    tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_l,
                                    decoder_max_length=3,
                                    batch_size=batch_s, shuffle=True, teacher_forcing=False, predict_eos_token=False,
                                    truncate_method="tail",num_workers=16,  # 根据CPU核心数调整
    pin_memory=True)  # 加速GPU数据传输)

validation_dataloader = PromptDataLoader(dataset=dataset["validation"], template=mytemplate, tokenizer=tokenizer,
                                         tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_l,
                                         decoder_max_length=3,
                                         batch_size=batch_s, shuffle=False, teacher_forcing=False,
                                         predict_eos_token=False,
                                         truncate_method="tail",num_workers=16,  # 根据CPU核心数调整
    pin_memory=True)



######################################################################################！！！！！！！！！！！！！！！！！！！！！！！！！！！

from transformers import  get_linear_schedule_with_warmup
from torch.optim import AdamW

# zero-shot test
test_dataloader = PromptDataLoader(dataset=dataset["test"], template=mytemplate, tokenizer=tokenizer,
                                   tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_l, decoder_max_length=3,
                                   batch_size=batch_s, shuffle=False, teacher_forcing=False, predict_eos_token=False,
                                   truncate_method="tail",num_workers=16,  # 根据CPU核心数调整
    pin_memory=True)
import csv
if args.verbalizer == "kpt" or args.verbalizer == "kpt_plus" :
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

    tot_step = len(train_dataloader) // args.gradient_accumulation_steps * args.max_epochs
    scheduler1 = get_linear_schedule_with_warmup(
        optimizer1,
        num_warmup_steps=0, num_training_steps=tot_step)

    optimizer2 = None
    scheduler2 = None

def evaluate1(prompt_model, dataloader, desc):
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

    # print(allpreds)
    # 写入 CSV 文件（每行一个值）
    with open(f'result/label_for_cal.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        for i in range(len(allpreds)):
            writer.writerow([alllabels[i], allpreds[i]])  # 每个值单独一行

    print("预测结果已写入 label_for_cal.csv")
    # 3. 读取原始测试数据（无列名）
    # df_test = pd.read_csv("datasets/TextClassification/rec-dy/test.csv", header=None)
    #
    # # 3. 验证数据匹配
    # if len(df_test) != len(allpreds):
    #     raise ValueError(f"数据量不匹配: 测试集{len(df_test)}行, 预测结果{len(allpreds)}个")
    #
    # # 4. 将预测结果写入test.csv的第7列（索引6）
    # df_test.iloc[:,7] = allpreds
    #
    #
    # # 5. 合并到训练集
    # if os.path.exists("datasets/TextClassification/rec-dy/train.csv"):
    #     df_train = pd.read_csv("datasets/TextClassification/rec-dy/train.csv", header=None)
    #     # 去重逻辑：根据文本列（假设是第0列）去重
    #     df_combined = pd.concat([df_train, df_test])
    #     df_merged = df_combined
    # else:
    #     df_merged = df_test
    #
    # # 6. 保存（不写入列名）
    # df_merged.to_csv("datasets/TextClassification/rec-dy/train.csv", index=False, header=False)
    # print(f"已合并 {len(df_test)} 行数据到训练集 | 最终行数: {len(df_merged)}")

    return









prompt_model.load_state_dict(torch.load(f"./ckpts/{this_run_unicode}.ckpt"))

prompt_model = prompt_model.cuda()

evaluate1(prompt_model, test_dataloader, desc="Test")




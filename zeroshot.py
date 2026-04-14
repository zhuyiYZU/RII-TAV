# from openprompt.utils.custom_tqdm import tqdm
from tqdm import tqdm
from openprompt.data_utils.text_classification_dataset import RelatedProcessor, AgnewsProcessor
import torch
from openprompt.data_utils.utils import InputExample
import argparse
import numpy as np

from openprompt import PromptDataLoader
from openprompt.prompts import ManualVerbalizer, KnowledgeableVerbalizer
from openprompt.prompts import ManualTemplate
import pandas as pd

parser = argparse.ArgumentParser("")
parser.add_argument("--shot", type=int, default=0)
parser.add_argument("--seed", type=int, default=144)

parser.add_argument("--plm_eval_mode", action="store_true")
parser.add_argument("--model", type=str, default='bert')
parser.add_argument("--model_name_or_path", default='model/bert_chinese')
# parser.add_argument("--model_name_or_path", default='/home/zy-4090-1/models/bert_base_cased')
parser.add_argument("--verbalizer", type=str)
parser.add_argument("--calibration", action="store_true")
parser.add_argument("--nocut", action="store_true")
parser.add_argument("--filter", default="none", type=str)
parser.add_argument("--template_id", type=int)
parser.add_argument("--max_token_split", default=-1, type=int)
parser.add_argument("--dataset", type=str)
parser.add_argument("--result_file", type=str, default="../sfs_scripts/results_zeroshot.txt")
parser.add_argument("--write_filter_record", action="store_true")
args = parser.parse_args()

from openprompt.utils.reproduciblity import set_seed

set_seed(args.seed)

from openprompt.plms import load_plm

plm, tokenizer, model_config, WrapperClass = load_plm(args.model, args.model_name_or_path)

dataset = {}

if args.dataset == "rec-related":
    dataset['train'] = RelatedProcessor().get_train_examples("datasets/veb/rec-related/")
    dataset['test'] = RelatedProcessor().get_test_examples("datasets/veb/rec-related/")
    class_labels = RelatedProcessor().get_labels()
    scriptsbase = "TextClassification/rec-related"
    scriptformat = "txt"
    cutoff = 0.5 if (not args.nocut) else 0.0
    max_seq_l = 128
    batch_s = 30
else:
    raise NotImplementedError

mytemplate = ManualTemplate(tokenizer=tokenizer).from_file(f"scripts/{scriptsbase}/manual_template.txt",
                                                           choice=args.template_id)

if args.verbalizer == "kpt":
    myverbalizer = KnowledgeableVerbalizer(tokenizer, classes=class_labels, candidate_frac=cutoff,
                                           max_token_split=args.max_token_split).from_file(
        f"scripts/{scriptsbase}/verbalizer.{scriptformat}")
elif args.verbalizer == "manual":
    myverbalizer = ManualVerbalizer(tokenizer, classes=class_labels).from_file(
        f"scripts/{scriptsbase}/manual_verbalizer.{scriptformat}")
elif args.verbalizer == "soft":
    raise NotImplementedError
elif args.verbalizer == "auto":
    raise NotImplementedError

# (contextual) calibration
if args.calibration:
    from openprompt.data_utils.data_sampler import FewShotSampler

    support_sampler = FewShotSampler(num_examples_total=200, also_sample_dev=False)
    dataset['support'] = support_sampler(dataset['train'], seed=args.seed)

    for example in dataset['support']:
        example.label = -1  # remove the labels of support set for clarification
    support_dataloader = PromptDataLoader(dataset=dataset["support"], template=mytemplate, tokenizer=tokenizer,
                                          tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_l,
                                          decoder_max_length=3,
                                          batch_size=batch_s, shuffle=False, teacher_forcing=False,
                                          predict_eos_token=False,
                                          truncate_method="tail", num_workers=16,  # 根据CPU核心数调整
                                          pin_memory=True)

from openprompt import PromptForClassification

use_cuda = True
prompt_model = PromptForClassification(plm=plm, template=mytemplate, verbalizer=myverbalizer, freeze_plm=False,
                                       plm_eval_mode=args.plm_eval_mode)
if use_cuda:
    prompt_model = prompt_model.cuda()

myrecord = ""
# HP
if args.calibration:
    org_label_words_num = [len(prompt_model.verbalizer.label_words[i]) for i in range(len(class_labels))]
    from openprompt.utils.calibrate import calibrate

    # calculate the calibration logits
    cc_logits = calibrate(prompt_model, support_dataloader)
    print("the calibration logits is", cc_logits)
    myrecord += "Phase 1 {}\n".format(org_label_words_num)

    myverbalizer.register_calibrate_logits(cc_logits)
    new_label_words_num = [len(myverbalizer.label_words[i]) for i in range(len(class_labels))]
    myrecord += "Phase 2 {}\n".format(new_label_words_num)

#
if args.write_filter_record:
    record_prefix = "=" * 20 + "\n"
    record_prefix += f"dataset {args.dataset}\t"
    record_prefix += f"temp {args.template_id}\t"
    record_prefix += f"seed {args.seed}\t"
    record_prefix += f"cali {args.calibration}\t"
    record_prefix += f"filt {args.filter}\t"
    record_prefix += "\n"
    myrecord = record_prefix + myrecord
    with open("../sfs_scripts/filter_record_file.txt", 'a') as fout_rec:
        fout_rec.write(myrecord)
    exit()

# zero-shot test
test_dataloader = PromptDataLoader(dataset=dataset["test"], template=mytemplate, tokenizer=tokenizer,
                                   tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_l, decoder_max_length=3,
                                   batch_size=batch_s, shuffle=False, teacher_forcing=False, predict_eos_token=False,
                                   truncate_method="tail", num_workers=16,  # 根据CPU核心数调整
                                   pin_memory=True)
allpreds = []
alllabels = []
pbar = tqdm(test_dataloader)
for step, inputs in enumerate(pbar):
    if use_cuda:
        inputs = inputs.cuda()
    logits = prompt_model(inputs)
    labels = inputs['label']
    alllabels.extend(labels.cpu().tolist())
    allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
acc = sum([int(i == j) for i, j in zip(allpreds, alllabels)]) / len(allpreds)
pd.DataFrame({"alllabels": alllabels, "allpreds": allpreds}).to_csv('result/out_label_dy.csv', header=0, index=False)
print(allpreds)
import pandas as pd

df1 = pd.read_csv(f'datasets/veb/{args.dataset}/test.csv', names=['text', 'label'])
df1['predict'] = allpreds
df2 = df1[['text', 'predict']]
df2.to_csv(f'datasets/veb/{args.dataset}/test.csv', index=False)
# roughly ~0.853 when using template 0


content_write = "=" * 20 + "\n"
content_write += f"dataset {args.dataset}\t"
content_write += f"temp {args.template_id}\t"
content_write += f"seed {args.seed}\t"
content_write += f"verb {args.verbalizer}\t"
content_write += f"cali {args.calibration}\t"
content_write += f"filt {args.filter}\t"
content_write += f"nocut {args.nocut}\t"
content_write += f"maxsplit {args.max_token_split}\t"
content_write += "\n"
content_write += f"Acc: {acc}"
content_write += "\n\n"

print(content_write)

with open(f"{args.result_file}", "a") as fout:
    fout.write(content_write)


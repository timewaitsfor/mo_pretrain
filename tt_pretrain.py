from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os
import logging
import argparse
import random
import pandas as pd
from tqdm import tqdm, trange
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler,Dataset
from torch.utils.data.distributed import DistributedSampler
import torch.nn as nn
from transformers import BertConfig, BertForSequenceClassification,LineByLineTextDataset,BertForMaskedLM, DataCollatorForLanguageModeling
from transformers import BertTokenizer,TrainingArguments,Trainer
from transformers import AdamW
import random
import time
import math

import utils.download_huggingface_models as dhf

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def dl_model(checkpoint):
    dhf.download_hf_models(checkpoint)


def main():
    import warnings
    warnings.filterwarnings("ignore")

    model_name = 'hfl/chinese-roberta-wwm-ext'
    model_dir = "./models/" + model_name.replace("/", "-")
    checkpoint = model_dir

    data_dir = './data/tt_text/train_pretrain_txt1229.txt'
    eval_data_dir = './data/tt_text/validation_pretrain_txt1229.txt'

    # dl_model(checkpoint) # 下载模型

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    random.seed(7580)
    np.random.seed(7580)
    torch.manual_seed(7580)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(7580)


    tokenizer = BertTokenizer.from_pretrained(checkpoint)
    bert_config = BertConfig.from_pretrained(checkpoint, hidden_dropout_prob=0.1)

    # Prepare model
    model = BertForMaskedLM(bert_config).from_pretrained(checkpoint)
    model.to(device)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
    train_dataset= LineByLineTextDataset(tokenizer = tokenizer,file_path = data_dir,block_size = 256)
    eval_dataset= LineByLineTextDataset(tokenizer = tokenizer,file_path = eval_data_dir,block_size = 256)
    # training_args = TrainingArguments(
    # output_dir='./models/chinese-roberta-wwm-ext_tmp1229',
    # overwrite_output_dir=True,
    # num_train_epochs=10,
    # learning_rate=2e-5,
    # per_device_train_batch_size=32,
    # save_total_limit=10,
    # save_steps=10000,
    # )
    # trainer = Trainer(
    # model=model,
    # args=training_args,
    # data_collator=data_collator,
    # train_dataset=train_dataset,
    # eval_dataset=eval_dataset,
    # )
    # trainer.train()
    # trainer.save_model('./models/chinese-roberta-wwm-ext_tmp1229_bp')
    #
    # eval_results = trainer.evaluate()
    # print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
    #
    # predictions = trainer.predict(eval_dataset)
    # print(predictions)
    #
    # preds = np.argmax(predictions.predictions, axis=-1)
    # print(preds)
    #
    # sequence1_ids = [[2260,5706,1524,1068,4010,5659,862,2138,3005,3013,3614,406,4085,5010,8013,2260]]
    # decode1_res = tokenizer.decode(sequence1_ids[0])
    # print(decode1_res)
    #
    # sequence2_ids = [[2260, 4212, 4339, 3300, 5774, 2647, 872, 2260, 511, 1557, 8024, 8024, 872, 872, 872, 1416]]
    # decode2_res = tokenizer.decode(sequence2_ids[0])
    # print(decode2_res)

    # print(eval_dataset)


    for idx, eval_d in enumerate(eval_dataset):
        if idx == 1:
            tmp_d = eval_d['input_ids']

    tmp_d = torch.unsqueeze(tmp_d, 0)
    tmp_d = tmp_d.to(device)
    print(tmp_d)

    attention_mask = torch.tensor([[1,1,0,1,1,1,1,1]])
    attention_mask = attention_mask.to(device)

    # outputs = model(tmp_d, attention_mask=attention_mask)
    # print(outputs)
    #
    # preds = np.argmax(outputs.predictions, axis=-1)
    # print(preds)

    # predicts = model.predict([tmp_d, attention_mask])[0]
    # print(predicts)

    # test_txt = torch.tensor("我真的好[MASK]你")

    test_txt = "巴黎是[MASK]国的首都。"
    # test_txt = torch.tensor(["巴黎是\[MASK\]国的首都。"])
    # test_txt = test_txt.to(device)

    # tokenizer.to(device)

    # model.cpu()
    from transformers import pipeline
    unmasker = pipeline('fill-mask', model=model, tokenizer=tokenizer, top_k=3)
    print(unmasker(test_txt))

if __name__ == "__main__":
    main()

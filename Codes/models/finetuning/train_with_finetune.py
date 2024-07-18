import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5,6"
import re
import rouge
import jieba
import torch
import argparse
import numpy as np
from tqdm.auto import tqdm
from bert4torch.models import *
from torch.utils.data import DataLoader, Dataset
import collections.abc as container_abcs
int_classes = int
string_classes = str
from transformers import MT5ForConditionalGeneration, BertTokenizer,T5Tokenizer
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    MBart50Tokenizer,
    MBart50TokenizerFast,
    MBartTokenizer,
    MBartTokenizerFast,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
    PegasusTokenizer,
    PegasusForConditionalGeneration,
    BertTokenizer,
    BartForConditionalGeneration,
    T5ForConditionalGeneration,
    MBartForConditionalGeneration,
)
from data_load import T5PegasusTokenizer, prepare_data
from compute_rouge import compute_rouges, compute_bleus, compute_meteros
from arguments import init_argument
from utils import get_logger, write_import_arsg_to_file, setup_seed
import evaluate

def train_model(model, adam, train_data, dev_data, test_data, tokenizer, device, args):
    model_save_path = f"{args.project_dir}/saved_models"
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    step = 0
    best = 0
    for epoch in range(args.num_epoch):
        model.train()
        for i, cur in enumerate(tqdm(train_data, desc='Epoch {}:'.format(epoch))):
            step += 1
            cur = {k: v.to(device) for k, v in cur.items()}

            prob = model(**cur)[0]
            mask = cur['decoder_attention_mask'][:, 1:].reshape(-1).bool()
            prob = prob[:, :-1]
            prob = prob.reshape((-1, prob.size(-1)))[mask]
            labels = cur['decoder_input_ids'][:, 1:].reshape(-1)[mask]
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(prob, labels)
            loss.backward()
            adam.step()
            adam.zero_grad()

            if step == 0:
                logger.warning(
                    "Iter {}:  Training Loss: {}".format(i, loss.item()))

            # 步数验证
            # if args.train_data_nums != -1:
            #     step_valid = args.train_data_nums // 100
            # else:
            #     step_valid = 100
            # if step % step_valid == 0:
            #     pass

        # 验证
        model.eval()
        gens = []
        summaries = []
        for feature in tqdm(dev_data):
            title = feature['title']
            content = {k: v.to(device)
                        for k, v in feature.items() if k != 'title'}
            if args.data_parallel and torch.cuda.is_available():
                gen = model.module. generate(max_length=args.max_len_generate,
                                                eos_token_id=tokenizer.sep_token_id,
                                                decoder_start_token_id=tokenizer.cls_token_id,
                                                **content)
            else:
                gen = model.generate(max_length=args.max_len_generate,
                                        eos_token_id=tokenizer.sep_token_id,
                                        decoder_start_token_id=tokenizer.cls_token_id,
                                        **content)
            gen = tokenizer.batch_decode(gen, skip_special_tokens=True)
            gen = [item.replace(' ', '') for item in gen]
            gens.extend(gen)
            summaries.extend(title)
        scores = compute_rouges(gens, summaries)
        bleus = compute_bleus(gens, summaries)
        bert_scores = compute_meteros(gens, summaries)
        logger.warning(
            "Epoch: {}, Step:{}, Validation Loss: {}".format(epoch, step, scores, bleus, bert_scores))
        rouge_l = scores['rouge-l']
        if rouge_l > best:
            best = rouge_l
            if args.data_parallel and torch.cuda.is_available():
                torch.save(model.module, os.path.join(
                    model_save_path, 'summary_model'))
                logger.critical("保存模型...")
            else:
                torch.save(model, os.path.join(
                    model_save_path, 'summary_model'))
                logger.critical("保存模型...")

            # test
            gens = []
            summaries = []
            
            for feature in tqdm(test_data):
                content = {k: v for k, v in feature.items() if k not in [
                    'dialog', 'title']}
                content = {k: v.to(device) for k, v in content.items()}
                if args.data_parallel and torch.cuda.is_available():
                    gen = model.module.generate(max_length=args.max_len_generate,
                                                eos_token_id=tokenizer.sep_token_id,
                                                decoder_start_token_id=tokenizer.cls_token_id,
                                                **content)
                else:
                    gen = model.generate(max_length=args.max_len_generate,
                                            eos_token_id=tokenizer.sep_token_id,
                                            decoder_start_token_id=tokenizer.cls_token_id,
                                            **content)
                gen = tokenizer.batch_decode(gen, skip_special_tokens=True)
                gen = [item.replace(' ', '') for item in gen]
                gens.extend(gen)
                if 'title' in feature:
                    summaries.extend(feature['title'])

            scores = compute_rouges(gens, summaries)
            bleus = compute_bleus(gens, summaries)
            bert_scores = compute_meteros(gens, summaries)
            logger.warning(
                "Epoch: {}, Step: {}, Test Metrics: {}, Bleu: {}, Meteros: {}".format(epoch, step, scores, bleus, bert_scores))
            print("Epoch: {}, Step: {}, Test Metrics: {}, Bleu: {}, Meteros: {}".format(epoch, step, scores, bleus, bert_scores))


if __name__ == '__main__':

    # step 1. init argument
    args = init_argument()
    logger = get_logger(args)

    # step 1.5 write_args_to_file
    write_import_arsg_to_file(logger=logger, args=args)
    setup_seed(args.seed)

    # step 2. prepare training data and validation data
    tokenizer = BertTokenizer.from_pretrained(args.pretrain_model)
    train_dataloader, valid_dataloader, test_dataloader = prepare_data(
        args, tokenizer)

    # step 3. load pretrain model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model =  BartForConditionalGeneration \
        .from_pretrained(args.pretrain_model).to(device)
    if args.data_parallel and torch.cuda.is_available():
        device_ids = range(torch.cuda.device_count())
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # step 4. finetune
    adam = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_model(model, adam, train_dataloader,
                valid_dataloader, test_dataloader, tokenizer, device, args)

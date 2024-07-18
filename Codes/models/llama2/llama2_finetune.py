import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
import sys
from typing import List

import fire
import torch
import transformers

from datasets import load_dataset
import time
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction
import nltk
import jieba
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
import rouge


def compute_metero(source, target):
    """计算两个中文句子的METEOR分数。"""
    # 使用jieba进行中文分词
    reference_tokenized = list(jieba.cut(source))
    candidate_tokenized = list(jieba.cut(target))
    try:
        # 计算METEOR分数
        score = meteor_score([reference_tokenized], candidate_tokenized)
        return {
            'meteor': score,
        }
    except ValueError as e:
        print(f"计算METEOR分数时出错: {e}")
        return {
            'meteor': 0.0,
        }

def compute_meteros(sources, targets):
    scores = {
        'meteor': 0.0,
    }
    for source, target in zip(sources, targets):
        score = compute_metero(source, target)
        for k, v in scores.items():
            scores[k] = v + score[k]

    return {k: v / len(targets) for k, v in scores.items()}

def compute_bleu(source, target):
    """计算两个中文句子的BLEU分数。"""
    # 使用jieba进行中文分词
    reference_tokenized = list(jieba.cut(source))
    candidate_tokenized = list(jieba.cut(target))
    chencherry = SmoothingFunction()
    try:
        # 计算BLEU分数
        score = sentence_bleu([reference_tokenized], candidate_tokenized,smoothing_function=chencherry.method5)
        return {
            'bleu': score,
        }
    except ValueError as e:
        print(f"计算BLEU分数时出错: {e}")
        return {
            'bleu': 0.0,
        }

def compute_bleus(sources, targets):
    scores = {
        'bleu': 0.0,
    }
    for source, target in zip(sources, targets):
        score = compute_bleu(source, target)
        for k, v in scores.items():
            scores[k] = v + score[k]

    return {k: v / len(targets) for k, v in scores.items()}

def compute_rouge(source, target):
    """计算rouge-1、rouge-2、rouge-l
    """
    source, target = ' '.join(source), ' '.join(target)
    try:
        scores = rouge.Rouge().get_scores(hyps=source, refs=target)
        return {
            'rouge-1': scores[0]['rouge-1']['f'],
            'rouge-2': scores[0]['rouge-2']['f'],
            'rouge-l': scores[0]['rouge-l']['f'],
        }
    except ValueError:
        return {
            'rouge-1': 0.0,
            'rouge-2': 0.0,
            'rouge-l': 0.0,
        }

def compute_rouges(sources, targets):
    scores = {
        'rouge-1': 0.0,
        'rouge-2': 0.0,
        'rouge-l': 0.0,
    }
    for source, target in zip(sources, targets):
        score = compute_rouge(source, target)
        for k, v in scores.items():
            scores[k] = v + score[k]

    return {k: v / len(targets) for k, v in scores.items()}


"""
Unused imports:
import torch.nn as nn
import bitsandbytes as bnb
"""
import json
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
    PeftModel,
)
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer

from prompter import Prompter


def train(
    # model/data params
    # base_model: str = "/home/jlchen/llama/llama2_7b",  # the only required argument
    base_model: str = "huggingface/Linly-AIChinese-LLaMA-2-7B-hf",  # your llama-2
    data_path: str = "",
    # output_dir: str = "./codswitch_emo2_2",
    # traindata_path: str = "./data/train_emo2.json",
    # devdata_path: str = "./data/dev_emo2.json",
    # testdata_path: str = './data/test_emo2.json',
    # result_path: str = "./csemo24090_2_1.txt",
    output_dir: str = "output_dir",
    traindata_path: str = "your train json",
    devdata_path: str = "your dev json",
    testdata_path: str = 'your test json',
    result_path: str = "res.txt",  #your result path
    # training hyperparams
    batch_size: int = 16,
    micro_batch_size: int = 4,
    #num_epochs: int = 50,
    num_epochs: int = 6,
    #num_epochs: int = 1,
    learning_rate: float = 3e-4,
    cutoff_len: int = 256,
    #val_set_size: int = 10,
    val_set_size: int = 800,
    # lora hyperparams
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    # lora_target_modules: List[str] = ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
    lora_target_modules: List[str] = [
        "q_proj",
        "v_proj",
        "k_proj",
        "o_proj",
        "gate_proj",
        "down_proj",
        "up_proj",
    ],
    # lora_target_modules: List[str] = [
    #     "q_proj",
    #     "v_proj",
    #     "k_proj",
    #     "o_proj",
    #     "gate_proj",
    #     "down_proj",
    # ],
    # llm hyperparams
    train_on_inputs: bool = True,  # if False, masks out inputs in loss
    add_eos_token: bool = False,
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    # wandb params
    wandb_project: str = "",
    wandb_run_name: str = "",
    wandb_watch: str = "",  # options: false | gradients | all
    wandb_log_model: str = "",  # options: false | true
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
    prompt_template_name: str = "alpaca",  # The prompt template to use, will default to alpaca.
):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Training Alpaca-LoRA model with params:\n"
            f"base_model: {base_model}\n"
            f"data_path: {data_path}\n"
            f"output_dir: {output_dir}\n"
            f"batch_size: {batch_size}\n"
            f"micro_batch_size: {micro_batch_size}\n"
            f"num_epochs: {num_epochs}\n"
            f"traindata_path: {traindata_path}\n"
            f"devdata_path: {devdata_path}\n"
            f"testdata_path: {testdata_path}\n"
            f"result_path: {result_path}\n"
            f"learning_rate: {learning_rate}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"val_set_size: {val_set_size}\n"
            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"add_eos_token: {add_eos_token}\n"
            f"group_by_length: {group_by_length}\n"
            f"wandb_project: {wandb_project}\n"
            f"wandb_run_name: {wandb_run_name}\n"
            f"wandb_watch: {wandb_watch}\n"
            f"wandb_log_model: {wandb_log_model}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
            f"prompt template: {prompt_template_name}\n"
        )

    gradient_accumulation_steps = batch_size // micro_batch_size

    prompter = Prompter(prompt_template_name)

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    # print('ddp', ddp)
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size
    # ddp = True
    # world_size = 2

    # if ddp:
    #     device_map = {"": 1}
    #     gradient_accumulation_steps = gradient_accumulation_steps // world_size
    # print(device_map)
    # print(gradient_accumulation_steps)
    # Check if parameter passed or if set within environ
    use_wandb = len(wandb_project) > 0 or (
        "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model

    model = LlamaForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map=device_map,
    )

    tokenizer = LlamaTokenizer.from_pretrained(base_model)

    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left"  # Allow batched inference
    print('device_map', device_map)
    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = prompter.generate_prompt(
            data_point["instruction"],
            data_point["input"],
            data_point["output"],
        )
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = prompter.generate_prompt(
                data_point["instruction"], data_point["input"]
            )
            tokenized_user_prompt = tokenize(
                user_prompt, add_eos_token=add_eos_token
            )
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            if add_eos_token:
                user_prompt_len -= 1

            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]  # could be sped up, probably
        return tokenized_full_prompt

    model = prepare_model_for_int8_training(model)

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    '''
    if data_path.endswith(".json") or data_path.endswith(".jsonl"):
        data = load_dataset("json", data_files=data_path)
    else:
        data = load_dataset(data_path)
    '''
    train_data = load_dataset("json", data_files=traindata_path)
    val_data = load_dataset("json", data_files=devdata_path)

    # test_data_path = ''
    # test_data = load_dataset("json", data_files=test_data_path)

    if resume_from_checkpoint:
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "adapter_model.bin"
            )  # only LoRA model - LoRA config above has to fit
            resume_from_checkpoint = (
                False  # So the trainer won't try loading its state
            )
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")

    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

    if val_set_size > 0:
        '''
        train_val = data["train"].train_test_split(
           test_size=val_set_size, shuffle=True, seed=42
        )
        '''

        train_data = (
            train_data["train"].shuffle().map(generate_and_tokenize_prompt)
        )
        val_data = (
           val_data["train"].shuffle().map(generate_and_tokenize_prompt)
        )
    #else:
    #    train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
    #    val_data = None
    #train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)


    if not ddp and torch.cuda.device_count() > 1:
        print('torch.cuda.device_count()', torch.cuda.device_count())
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=10,
            optim="adamw_torch",
            evaluation_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=200 if val_set_size > 0 else None,
            save_steps=200,
            #sharded_ddp="simple",
            output_dir=output_dir,
            save_total_limit=3,
            load_best_model_at_end=True if val_set_size > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to="wandb" if use_wandb else None,
            run_name=wandb_run_name if use_wandb else None,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    model.config.use_cache = False

    #old_state_dict = model.state_dict
    #model.state_dict = (
       #lambda self, *_, **__: get_peft_model_state_dict(
            #self, old_state_dict()
        #)
    #).__get__(model, type(model))

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)
    print('before train', time.ctime())
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    print('after train', time.ctime())

    model.save_pretrained(output_dir)

    print(
        "\n If there's a warning about missing keys above, please disregard :)"
    )


    def evaluate(
        instruction,
        input=None,
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=2,
        max_new_tokens=128,
        stream_output=False,
        **kwargs,
    ):
        prompt = prompter.generate_prompt(instruction, input)
        inputs = tokenizer(prompt, return_tensors="pt")
        device = 'cuda'
        input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )

        # Without streaming
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        s = generation_output.sequences[0]
        output = tokenizer.decode(s)
        # return output
        return prompter.get_response(output)

    model.eval()

    '''test_data_path = ''
    test_data = load_dataset("json", data_files=test_data_path)

    for input in test_data['train']:
        instruction = input['instruction']
        input_sentence = input['input']
        output = evaluate(instruction=instruction, input=input_sentence)
        print(output)

    import json
    with open('newtest.json') as f:
        testline = json.load(f)'''
    #import json
    with open(testdata_path) as f:
        testline = json.load(f)
    ROUGE_1 = 0
    ROUGE_2 = 0
    ROUGE_L = 0
    BLEU = 0
    METEOR = 0
    # out=open("predlabel.txt","w",encoding="utf-8")
    out1 = open(result_path, "a", encoding="utf-8")
    number = 0
    for test in testline:
        instruction = test['instruction']
        input = test['input']
        output = evaluate(instruction=instruction, input=input)
        reference = test['output']
        ROUGE = compute_rouge(reference, output)
        bleu = compute_bleu(reference, output)
        meteor = compute_metero(reference, output)
        ROUGE_1 += ROUGE['rouge-1']
        ROUGE_2 += ROUGE['rouge-2']
        ROUGE_L += ROUGE['rouge-l']
        BLEU += bleu['bleu']
        METEOR += meteor['meteor']
        number += 1
        print('ROUGE_1', ROUGE_1/number, 'ROUGE_2', ROUGE_2/number, 'ROUGE_L', ROUGE_L/number, 'BLEU', BLEU/number, 'METEOR', METEOR/number)
        print('ROUGE_1', ROUGE_1/number, 'ROUGE_2', ROUGE_2/number, 'ROUGE_L', ROUGE_L/number, 'BLEU', BLEU/number, 'METEOR', METEOR/number, file=out1)
    ROUGE_1 = ROUGE_1/number
    ROUGE_2 = ROUGE_2/number
    ROUGE_L = ROUGE_L/number
    BLEU = BLEU/number
    METEOR = METEOR/number
    print('ROUGE_1', ROUGE_1, 'ROUGE_2', ROUGE_2, 'ROUGE_L', ROUGE_L, 'BLEU', BLEU, 'METEOR', METEOR,file=out1)
    print('ROUGE_1', ROUGE_1, 'ROUGE_2', ROUGE_2, 'ROUGE_L', ROUGE_L, 'BLEU', BLEU,'METEOR', METEOR)

    # count = 0
    # output = ''
    # dic = {"Chinese and English":"中文和英文","Chinese":"中文","English":"英文"}
    # s = {"Chinese and English","Chinese","English"}
    # #import json
    # with open(testdata_path) as f:
    #     testline = json.load(f)
    #
    # # out=open("predlabel.txt","w",encoding="utf-8")
    # out1 = open(result_path, "w", encoding="utf-8")
    # for test in testline:
    #     if count%2 == 1:
    #         # if output not in s:
    #         #     output = "中文和英文"
    #         # else:
    #         #     output = dic[output]
    #         # instruction = "这句话用"+output+"表达情绪,这句话的情绪是什么"
    #         instruction = "This sentence use "+output+" to express emotion, what is the emotion of this sentence"
    #     else:
    #         instruction = test['instruction']
    #     input = test['input']
    #     output = evaluate(instruction=instruction, input=input)
    #     output = list(output)
    #     output = ''.join(output)
    #     result = output.split("\n\n")[0]
    #
    #     print(result)
    #     print(result, file=out)'''
    #     count += 1
    #     print(output)
    #     print(output,file=out1)


if __name__ == "__main__":
    train()

import torch
import os
import csv
import logging
from tqdm.auto import tqdm
from transformers import MT5ForConditionalGeneration
from data_load import T5PegasusTokenizer, prepare_data
from compute_rouge import compute_rouges
from arguments import init_argument
from utils import get_logger, write_import_arsg_to_file, setup_seed


def generate(test_data, model, tokenizer, result_file, args):
    gens, summaries = [], []
    with open(result_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        model.eval()
        for feature in tqdm(test_data):
            raw_data = feature['dialog']
            content = {k: v for k, v in feature.items() if k not in [
                'dialog', 'title']}
            content = {k: v.to(device) for k, v in content.items()}
            gen = model.generate(max_length=args.max_len_generate,
                                 eos_token_id=tokenizer.sep_token_id,
                                 decoder_start_token_id=tokenizer.cls_token_id,
                                 **content)
            gen = tokenizer.batch_decode(gen, skip_special_tokens=True)
            gen = [item.replace(' ', '') for item in gen]
            writer.writerows(zip(gen, raw_data))
            gens.extend(gen)
            if 'title' in feature:
                summaries.extend(feature['title'])
    if summaries:
        scores = compute_rouges(gens, summaries)
        logger.info(f"{scores}")
        print(scores)
    print('Done!')


def generate_test(test_data, model, tokenizer, args):
    model.eval()
    for feature in tqdm(test_data):

        content = {k: v for k, v in feature.items() if k not in [
            'dialog', 'title']}
        content = {k: v.to(device) for k, v in content.items()}
        gen = model.generate(max_length=args.max_len_generate,
                             eos_token_id=tokenizer.sep_token_id,
                             decoder_start_token_id=tokenizer.cls_token_id,
                             **content)
        gen = tokenizer.batch_decode(gen, skip_special_tokens=True)

        print(gen)


if __name__ == '__main__':

    # step 1. init argument
    args = init_argument()

    # step 2. prepare test data
    tokenizer = T5PegasusTokenizer.from_pretrained(args.pretrain_model)
    test_data = prepare_data(args, tokenizer, Only_test=True)

    #
    results_path = "/opt/data/private/nlp03/kdwang/projects/JDZY/src/results/more_tokens_200_tokens"
    log_name = f"{results_path}/test.log"
    logging.basicConfig(level=logging.DEBUG,
                        filename=log_name,
                        datefmt='%Y/%m/%d %H:%M:%S',
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger()

    # step 3. load finetuned model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # for i in range(2600, 3700, 100):
    #     logger.info(f"dataset length: {i}")
    #     result_file = f"{results_path}/test_metrics/predict_result_{i}.tsv"
    #     model = torch.load(
    #         f"{results_path}/{str(i)}_dialogs/summary_model", map_location=device)
    #     generate(test_data, model, tokenizer, result_file, args)

    result_file = f"/opt/data/private/nlp03/kdwang/projects/JDZY/src/results/few_shot_250_tokens/0_dialogs/predict_result_0.tsv"
    model = MT5ForConditionalGeneration.from_pretrained(
        "/opt/data/private/nlp03/kdwang/huggingface_models/t5-pegasus-base").to(device)
    generate(test_data, model, tokenizer, result_file, args)

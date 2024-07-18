import rouge
import bleu
from bert_score import score as btscore
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
import jieba

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


def compute_bert_score(source, target):
    """计算bert_score
    """
    model_path = '/home/ycshi/huggingface/bert-base-chinese'
    source, target = ' '.join(source), ' '.join(target)
    try:
        P, R, F1 = btscore([source], [target], model_type=model_path, lang='zh', verbose=False)
        return {
            'bert_score': F1.mean().item(),
        }
    except ValueError:
        return {
            'bert_score': 0.0,
        }
        
def compute_bert_scores(sources, targets):
    scores = {
        'bert_score': 0.0,
    }
    for source, target in zip(sources, targets):
        score = compute_bert_score(source, target)
        for k, v in scores.items():
            scores[k] = v + score[k]

    return {k: v / len(targets) for k, v in scores.items()}
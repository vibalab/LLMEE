
# PyTorch 및 모델 관련 라이브러리
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import transformers

# BitsandBytes 라이브러리
import bitsandbytes as bnb

# SafeTensor 파일 작업
from safetensors import safe_open

# 웹 스크레이핑 라이브러리
import requests
from lxml import html

# 평가 라이브러리
import evaluate
import nltk.translate.bleu_score as bleu
from rouge_score import rouge_scorer
from huggingface_hub import HfApi

# 해석 및 설명 관련 라이브러리
from captum.attr import (
    FeatureAblation, 
    ShapleyValues,
    LayerIntegratedGradients, 
    LLMAttribution, 
    LLMGradientAttribution, 
    TextTokenInput, 
    TextTemplateInput,
    ProductBaselines,
    ShapleyValueSampling,
    Lime,
    KernelShap,
    LLMAttributionResult
)
import shap

import re

# 시스템 및 유틸리티 라이브러리
import random
import sys
import os

access_token = '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'

def load_model(model_name, bnb_config):
    # n_gpus = torch.cuda.device_count()
    # max_memory = "10000MB"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto", # dispatch efficiently the model on the available ressources
        # max_memory = {i: max_memory for i in range(n_gpus)},
        token=access_token 
    )

    # model.config.task_specific_params = dict()

    # # 텍스트 생성 파라미터 설정
    # gen_args = dict(
    #     max_new_tokens=10,
    #     num_beams=5,
    #     renormalize_logits=True,
    #     # do_sample=False,           # 문장 고정
    #     # temperature=0.7,           # 문장 고정
    #     # top_k=50,                  # 문장 고정
    #     no_repeat_ngram_size=2       # SHAP default
    # )

    # model.config.task_specific_params["text-generation"] = gen_args
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=access_token)

    # Needed for LLaMA tokenizer
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer

def create_bnb_config():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,  # 4비트 양자화 사용
        bnb_4bit_use_double_quant=True,  # 더블 양자화 활성화
        bnb_4bit_quant_type="nf4",  # 4비트 양자화 유형 설정
        bnb_4bit_compute_dtype=torch.bfloat16,  # 연산 시 데이터 타입 설정
        load_in_8bit_fp32_cpu_offload=True  # 일부 모듈을 32비트로 CPU에 오프로딩
    )

    return bnb_config



def ensure_list(variable):
    if isinstance(variable, str):
        return [variable]  # If it's a string, convert it to a list
    elif isinstance(variable, list):
        return variable  # If it's already a list, return as is
    else:
        raise TypeError(f"TypeError: Expected a list or string, but got {type(variable)}")

def compute_attributions(model, tokenizer, eval_prompt, explanation_method, target):
    # Apply the function to both eval_prompt and target
    eval_prompt = ensure_list(eval_prompt)
    target = ensure_list(target)
            
    model.eval()

    # Initialize the selected explanation method
    if explanation_method == 'FeatureAblation':
        explainer = FeatureAblation(model)
    
    elif explanation_method == 'SHAP':
        # Ver.1
        # explainer = ShapleyValues(model)  # This can be computationally expensive
        
        # Ver.2
        shap_model = shap.models.TeacherForcing(model, tokenizer) 

        masker = shap.maskers.Text(tokenizer, mask_token="...", collapse_mask_token=True)
        explainer = shap.Explainer(shap_model, masker)

        shap_values = explainer(eval_prompt, target)

        attr=torch.Tensor(shap_values.values).T

        seq_attr = attr.sum(0).squeeze()  

        # return shap_values
        return LLMAttributionResult(
            seq_attr, 
            attr, 
            input_tokens=shap_values.feature_names[0], 
            output_tokens=shap_values.output_names
            )
    
    elif explanation_method == 'ShapleyValueSampling':
        explainer = ShapleyValueSampling(model)  # An approximation
    
    # Lime and KernelShap do not support per-token attribution and will only return attribution for the full target sequence.
    elif explanation_method == 'LIME':
        explainer = Lime(model)
    
    elif explanation_method == 'KernelSHAP':
        explainer = KernelShap(model)
    
    else:
        raise ValueError(f"Unknown explanation method: {explanation_method}")

    # Initialize the LLM attribution wrapper
    # llm_attr = LLMAttribution(explainer, tokenizer, attr_target='prob') #attr_target = ['log_prob','prob']
    llm_attr = LLMAttribution(explainer, tokenizer, attr_target='log_prob') #attr_target = ['log_prob','prob']

    # Prepare the input for the model
    inp = TextTokenInput(
        eval_prompt[0], 
        tokenizer,
        skip_tokens=[1],  # skip the special token for the start of the text <s>
    )    

    # target = "playing guitar, hiking, and spending time with his family."
    # # target response with respect to which attributions are computed. 
    # # If None, it uses the model to generate the target based on the input and gen_args. Default: None

    # Compute the attributions
    attr_res = llm_attr.attribute(inp, target = target[0])
    
    # 모델마다 token이 달라서 다른 방법을 찾아야 할 듯
    attr_res.input_tokens = list(map(lambda x: x.replace("Ġ", ""), attr_res.input_tokens)) 
    attr_res.output_tokens = list(map(lambda x: x.replace("Ġ", ""), attr_res.output_tokens))
    
    return attr_res

def generate_text_prob(model, tokenizer, eval_prompt, target):
    eval_prompt = ensure_list(eval_prompt)
    target = ensure_list(target)

    model_input = tokenizer(eval_prompt, return_tensors="pt").to("cuda")
    model.eval()
    
    with torch.no_grad():
        output_ids = model.generate(model_input["input_ids"], max_new_tokens=50, do_sample=False)[0]
        response = tokenizer.decode(output_ids, skip_special_tokens=True)
        input_text = tokenizer.decode(model_input["input_ids"][0], skip_special_tokens=True)
        gen_text = response[len(input_text):].strip()

    gen_text = gen_text.strip()

    sentences = re.split(r'(?<=[.!?])\s+', gen_text)

    if sentences:  # 문장이 존재할 때만
        sentence = sentences[0]

    # BLEU    
    # score = bleu.sentence_bleu(list(map(lambda ref: ref.split(), [gen_text])),target[0].split())
    
    # ROUGE
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_score = scorer.score(gen_text, target[0])
    
    # BLEURT: 사전 학습된 언어 모델을 기반으로 하여 인간의 평가 점수를 모방
    metric = evaluate.load("bleurt", trust_remote_code=True)
    bleurt_score = metric.compute(predictions=target, references=[gen_text])['scores'][0]
    
    return sentence, rouge_score, bleurt_score

def get_model_info(model_name):
    base_url = "https://huggingface.co"
    url = f"{base_url}/{model_name}"

    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to retrieve page. Status code: {response.status_code}")
        return None, None

    # lxml로 HTML 파싱
    tree = html.fromstring(response.content)

    # XPath를 사용하여 요소 선택
    model_size = tree.xpath('/html/body/div/main/div[2]/section[2]/div[3]/div/div[2]/div[1]/div[2]/text()')
    tensor_type = tree.xpath('/html/body/div/main/div[2]/section[2]/div[3]/div/div[2]/div[2]/div[2]/text()')

    # XPath는 리스트를 반환하므로, 첫 번째 요소를 가져옵니다.
    model_size = model_size[0].strip() if model_size else None
    tensor_type = tensor_type[0].strip() if tensor_type else None

    return model_size, tensor_type

def get_model_card(model_name, model):
    model_card = {}
    api = HfApi()
    model_info = api.model_info(model_name)
    model_card['language'] = model_info.cardData.get('language', 'en')
    if isinstance(model_card['language'], str):
        model_card['language'] = [model_card['language']]

    model_card['tags'] = model_info.cardData.get('tags', None)

    model_card['model_size'], model_card['tensor_type'] = get_model_info(model_name)
    
    config = model.config
    
    model_card['model_type'] = str(getattr(config, 'model_type', None))
    model_card['vocab_size'] = getattr(config, 'vocab_size', None)

    return model_card

def update_dict(model_dict, tokenizer):
    # Special tokens 리스트 가져오기
    special_tokens = tokenizer.all_special_tokens

    input_tokens = model_dict["input_tokens"]

    # Special token이 있는 인덱스 추적
    indices_to_remove = [i for i, token in enumerate(input_tokens) if token in special_tokens]

    # Special tokens 제거한 토큰 리스트 생성
    cleaned_tokens = [token for token in input_tokens if token not in special_tokens]

    model_dict["input_tokens"] = remove_special_prefixes(cleaned_tokens)

    model_dict["output_tokens"] = remove_special_prefixes(model_dict["output_tokens"])

    # seq_attr에서 해당 인덱스를 제거 (1차원 배열이므로 해당 인덱스에서 값 삭제)
    model_dict['seq_attr'] = [attr for i, attr in enumerate(model_dict['seq_attr']) if i not in indices_to_remove]
    
    # token_attr에서 해당 인덱스를 제거 (2차원 배열이므로 각 열에서 해당 인덱스 삭제)
    if model_dict['token_attr'] != None:
        model_dict['token_attr'] = [[attr for i, attr in enumerate(row) if i not in indices_to_remove] for row in model_dict['token_attr']]

    return model_dict


def remove_special_prefixes(tokens):
    # 제거할 특수 문자 리스트
    special_prefixes = ['##', 'Ġ', '▁', '_']
    
    # 특수 문자 제거한 토큰들
    processed_tokens = []
    for token in tokens:
        for prefix in special_prefixes:
            if token.startswith(prefix):
                token = token[len(prefix):]
                break  # 한 번 특수 문자를 제거했으면 다음으로 넘어감
        processed_tokens.append(token)
    return processed_tokens


import bitsandbytes as bnb
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import random
import sys

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
    ShapleyValues,
    Lime,
    KernelShap,
    LLMAttributionResult
)

import transformers

import shap
import torch

def load_model(model_name, bnb_config):
    # n_gpus = torch.cuda.device_count()
    # max_memory = "10000MB"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto", # dispatch efficiently the model on the available ressources
        # max_memory = {i: max_memory for i in range(n_gpus)},
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
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)

    # Needed for LLaMA tokenizer
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer

def create_bnb_config():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    return bnb_config



def compute_attributions(model, tokenizer, eval_prompt, explanation_method, target = None):
    if not target:
        model_input = tokenizer(eval_prompt, return_tensors="pt").to("cuda")
        model.eval()
        
        with torch.no_grad():
            output_ids = model.generate(model_input["input_ids"], max_new_tokens=15, do_sample=False)[0]
            response = tokenizer.decode(output_ids, skip_special_tokens=True)
            input_text = tokenizer.decode(model_input["input_ids"][0], skip_special_tokens=True)
            target = [response[len(input_text):].strip()]

    # Initialize the selected explanation method
    if explanation_method == 'FeatureAblation':
        explainer = FeatureAblation(model)
    
    elif explanation_method == 'ShapleyValues':
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
    elif explanation_method == 'Lime':
        explainer = Lime(model)
    
    elif explanation_method == 'KernelShap':
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
    # attr_res.input_tokens = list(map(lambda x: x.replace("Ġ", ""), attr_res.input_tokens)) 
    # attr_res.output_tokens = list(map(lambda x: x.replace("Ġ", ""), attr_res.output_tokens))
    
    return attr_res


import os
import threading
from queue import Queue
from flask import Flask, request, jsonify, render_template
from transformers import AutoModelForCausalLM, AutoTokenizer
from flask_cors import CORS
from transformers import AutoModelForImageClassification
import torch
import numpy as np
from attr_contribute import get_model_card, create_bnb_config, load_model, compute_attributions, generate_text_prob, update_dict
import json
import gc
import huggingface_hub
import socket

app = Flask(__name__)
CORS(app)

# Determine number of GPUs and create a queue for inference tasks
num_gpus = torch.cuda.device_count()
inference_queue = Queue()
loaded_models = []
language_model = None
tokenizer = None

@app.route('/')
def home():
    return render_template('index.html')
    
def _unload_model():
    global language_model, tokenizer
    if language_model is not None:
        # language_model.to('cpu')
        del language_model
        torch.cuda.empty_cache()  # GPU 메모리 캐시 해제
        gc.collect()
    language_model = None
    tokenizer = None

def _is_valid_explanation_method(method):
    # Valid explanation methods
    valid_methods = ['FeatureAblation', 'ShapleyValues', 'ShapleyValueSampling', 'LIME', 'KernelSHAP', 'SHAP']
    return method in valid_methods

@app.route('/loadModel', methods=['POST'])
def model():
    global language_model, tokenizer
    _unload_model()
    model_list = json.loads(request.form.get('model_list'))
    
    model_directory = './models'

    if not os.path.exists(model_directory):
        os.makedirs(model_directory) 

    loaded_models = []

    for model_name in model_list:
        local_model_path = os.path.join(model_directory, model_name)  
        
        if not os.path.exists(local_model_path) or len(os.listdir(local_model_path)) == 0:
            if not os.path.exists(local_model_path):
                os.makedirs(local_model_path) 
            
            #If model is not downloaded, download model to './models/${model_name}'
            bnb_config = create_bnb_config()
            language_model, tokenizer = load_model(model_name, bnb_config)

            language_model.save_pretrained(local_model_path + "/")
            tokenizer.save_pretrained(local_model_path + "/")

            _unload_model()
                    
            # except ValueError as e:
            #     # Invalid model_name
            #     return jsonify({"status": "error", "message": f"ValueError: {str(e)}"}), 400
            
            # except Exception as e:
            #     # Other errors
            #     return jsonify({"status": "error", "message": f"An unexpected error occurred: {str(e)}"}), 500
        
        loaded_models.append(model_name)

    return jsonify({"message": "All models are loaded", "loaded_models": loaded_models})

@app.route('/input', methods=['POST'])
def inputs():
    global language_model, tokenizer
    _unload_model()
    input_prompt = request.form.get('input')
    output_prompt = request.form.get('output')
    xai_method = request.form.get('xai-method').replace(" ", "")
    model_list = json.loads(request.form.get('model_list'))

    results = {}

    # Create the configuration for the model
    bnb_config = create_bnb_config()

    for model_name in model_list:
        # Model path
        model_dir = os.path.join("./models", model_name)
        
        language_model, tokenizer = load_model(model_dir, bnb_config)
        attribute = compute_attributions(language_model, tokenizer, input_prompt, xai_method, output_prompt)

        model_dict = {}
                
        model_dict["token_attr"] = getattr(attribute, 'token_attr', None)
        if model_dict['token_attr'] != None:
            model_dict['token_attr'] = model_dict['token_attr'].tolist(),  # token_attr가 없을 경우 None 반환
        model_dict["seq_attr"] = attribute.seq_attr.tolist()          

        model_dict["input_tokens"] = attribute.input_tokens    
        model_dict["output_tokens"] = attribute.output_tokens

        model_dict = update_dict(model_dict, tokenizer)
        # #If user doesn't select XAI method 
        # except Exception as e:
        #     return jsonify({"status": "error", "message": f"Failed to generate text for model {model_name}: {str(e)}"}), 500

        # Check if explanation method is valid
        if not _is_valid_explanation_method(xai_method):
            return jsonify({"status": "error", "message": "Invalid explanation method."}), 400

        model_dict["generated_text"], model_dict["rouge_score"], model_dict["bleurt_score"] = generate_text_prob(language_model, tokenizer, input_prompt, output_prompt)
        model_dict["model_card"] = get_model_card(model_name, language_model)
        
        results[model_name] = model_dict

        _unload_model()
    
    _unload_model()

    return jsonify({"status": "success", "results": results})

if __name__ == '__main__':
    app.run(port='5011', debug=True)

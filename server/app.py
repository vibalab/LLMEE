import os
import threading
from queue import Queue
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from transformers import AutoModelForImageClassification
import torch
import numpy as np
from attr_contribute import create_bnb_config, load_model, compute_attributions

app = Flask(__name__)
CORS(app)

# Determine number of GPUs and create a queue for inference tasks
num_gpus = torch.cuda.device_count()
inference_queue = Queue()

@app.route('/')
def home():
    return render_template('index.html')
    

def _is_valid_explanation_method(method):
    # Valid explanation methods
    valid_methods = ['FeatureAblation', 'ShapleyValues', 'ShapleyValueSampling', 'Lime', 'KernelShap']
    return method in valid_methods
    
@app.route('/attribute', methods=['POST'])
def attribute():
    # Receive data from client
    model_name = request.form.get('model_name')
    input_prompt = request.form.get('input_prompt')             
    target_text = request.form.get('target_text')               # If target_text is None, we generate target text automatically.
    explanation_method = request.form.get('explanation_method') # FeatureAblation, ShapleyValues, ShapleyValueSampling, Lime, KernelShap
    
    
    # Check if explanation_method is valid
    if not _is_valid_explanation_method(explanation_method):
        return jsonify({"status": "error", "message": "Invalid explanation method."}), 400
    
    # Create the configuration for the model
    bnb_config = create_bnb_config()
    
    try:
        #Load model
        model, tokenizer = load_model(model_name, bnb_config)

    except ValueError as e:
        #Invalid model_name
        return jsonify({"status": "error", "message": f"ValueError: {str(e)}"}), 400
    
    except Exception as e:
        #Other errors
        return jsonify({"status": "error", "message": f"An unexpected error occurred: {str(e)}"}), 500
    
    result = compute_attributions(model, tokenizer, input_prompt, explanation_method, target_text)

    return jsonify({
        "input_text": result.input_tokens,
        "output_tokens": result.output_tokens,
        "token_attr": getattr(result, 'token_attr', None),  # token_attr가 없을 경우 None 반환
        "seq_attr": result.seq_attr
    })
if __name__ == '__main__':
    app.run(debug=True)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from flask import Flask, request, jsonify
from time import time
from transformers import pipeline
from functools import lru_cache

app = Flask(__name__)

MODEL_PATH = "bert_multilingual_model_app_review"
# Perform model loading once, return loaded model from cache for subsequent predict calls
@lru_cache()
def load_model(model_path=MODEL_PATH):
    start = time()
    sentiment_classifier = pipeline(task="sentiment-analysis",
                                    model=model_path)
    elapsed = time() - start
    print('Model loaded in {} seconds'.format(elapsed))
    return sentiment_classifier

@app.route('/sentiment/', methods=['POST'])
def get_sentiment():
    start = time()
    input_text = request.args.get("text", "")
    sentiment_classifier = load_model(MODEL_PATH)
    sentiment = sentiment_classifier(input_text)
    elapsed = time() - start
    print('Inference time: {}'.format(elapsed))
    return jsonify(sentiment)


if __name__ == '__main__':
    load_model()
    app.run(host='0.0.0.0', port=5000)
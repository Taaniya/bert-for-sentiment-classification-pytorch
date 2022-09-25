# BERT for sentiment classification using pytorch
This repository contains notebooks &amp; python scripts for fine tuning BERT for sentiment classification task

### Run model training
```
python train.py --model_name "bert-base-multilingual-cased" 
--seed 42 --batch_size 64 
--model_dir /path/to/dir/to/save/model 
--dataset_dir /path/to/load/dataset
```

### References
1. https://www.curiousily.com/posts/sentiment-analysis-with-bert-and-hugging-face-using-pytorch-and-python/
2. https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
3. https://mccormickml.com/2019/07/22/BERT-fine-tuning/
4. https://pytorch.org/docs/stable/data.html#loading-batched-and-non-batched-data
5. https://pytorch.org/docs/stable/notes/amp_examples.html
6. https://developer.nvidia.com/blog/mixed-precision-training-deep-neural-networks/
7. https://pytorch.org/blog/accelerating-training-on-nvidia-gpus-with-pytorch-automatic-mixed-precision/


#### Run sentiment detection service in docker container
```
# Build image
docker build -t bert-sentiment-classifier . 

# Run it in a container in detached mode
docker run -d -p 5000:5000 bert-sentiment-classifier
```

##### Test sentiment with curl from local machine

```
curl --location --request POST 'http://localhost:5000/sentiment' \
--header 'Content-Type: text/plain' \
--data-raw '{
    "text":"the user interface was quite easy to use"
}'
```
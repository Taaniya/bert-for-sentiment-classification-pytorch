# -*- coding: utf-8 -*-

"""
This script fine tunes pretrained BERT model on play store app reviews for sentiment classification 
using app ratings as labels.
"""

import datetime
import time
import random
import os
import argparse

import pandas as pd
import numpy as np

from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast

# Defining custom Dataset class by inheriting Dataset abstract class from torch and 
# overriding __len__ & __getitem__ functions

class ReviewDataset(Dataset):
  """ Custom Dataset class. """

  def __init__(self, reviews, targets, tokenizer, max_len):
    self.reviews = reviews
    self.targets = targets
    self.tokenizer = tokenizer
    self.max_len = max_len

  def __len__(self):
    return len(self.reviews)

  def __getitem__(self, idx):
    """ Method to support indexing and return dataset[idx] """
    review = str(self.reviews[idx])
    target = self.targets[idx]
    encoding = self.tokenizer.encode_plus(review, 
                                          add_special_tokens=True,
                                          max_length=self.max_len,
                                          return_attention_mask=True,
                                          return_tensors='pt',
                                          return_token_type_ids=False,
                                          pad_to_max_length=True,
                                          truncation=True)
    return {
      'review_text' : review,
      'input_ids' : encoding['input_ids'].flatten(),
      'attention_mask' : encoding['attention_mask'].flatten(),
      'targets' : torch.tensor(target, dtype=torch.long)
    }

def create_data_loader(df, tokenizer, max_len, batch_size):
  ds = ReviewDataset(
    reviews=df['content'].to_numpy(),
    targets=df['label'].to_numpy(),
    tokenizer=tokenizer,
    max_len=max_len
  )
  # shuffle true is recommended so that batched between epochs donot look alike
  return DataLoader(
    ds,
    batch_size=batch_size,
    num_workers=4,
    shuffle=True,
    pin_memory=True  # For faster data transfer from host to GPU in CUDA-enabled GPUs   
  )

def flat_accuracy(preds, labels):
  """ Function to calculate accuracy of our predictions vs labels 
  """
  pred_flat = np.argmax(preds, axis=1).flatten()
  labels_flat = labels.flatten()
  return np.sum(pred_flat == labels_flat) / len(labels_flat)


def format_time(elapsed):
    '''Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

def main():
    parser = argparse.ArgumentParser(description="BERT training for sentiment classification for 2 classes", add_help=True)
    parser.add_argument("--model_name", type=str, default="bert-base-multilingual-cased", help="pre-trained model name or path")
    parser.add_argument("--seed", type=int, default=42 , help="random seed value")
    parser.add_argument("--batch_size", type=int, required=True, help="batch size for training")
    parser.add_argument("--max_len", type=int, default=100, help="Maximum length of sequence")
    parser.add_argument("--epoch", type=int, default=1 , help="No. of training epochs")
    parser.add_argument("--model_dir", type=str, required=True, help="path to directory to save model")
    parser.add_argument("--lr", type=float, default=2e-5, help="learning rate")
    parser.add_argument("--dataset_dir", type=str, required=True, help="path to directory to load datasets")
    args = parser.parse_args()

    RANDOM_SEED = args.seed
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    PRETRAINED_MODEL_NAME = args.model_name
    MAX_LEN = args.max_len                 # maximum length of sequence after tokenization
    BATCH_SIZE = args.batch_size
    lr = args.lr
    EPOCHS = args.epoch                    # 3-4 epochs suffice for fine-tuning BERT
    eps = 1e-8

    # Load data
    drive_path = args.dataset_dir
    df_train = pd.read_csv(drive_path+"train.csv")
    df_val = pd.read_csv(drive_path+"validation.csv")
    df_test = pd.read_csv(drive_path+"test.csv")

    model_dir = args.model_dir
    model_save_path = model_dir + "bert_multilingual_model_app_review/"

    # Initializing model based tokenizer
    tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME, do_lower_case=False)

    train_data_loader = create_data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE)
    val_data_loader = create_data_loader(df_val, tokenizer, MAX_LEN, BATCH_SIZE)
    test_data_loader = create_data_loader(df_test, tokenizer, MAX_LEN, BATCH_SIZE)

    # Using BertforSequenceClassification which is same as adding a linear layer to pre-trained BertModel which 
    # returns raw hidden states having no head on top
    model = BertForSequenceClassification.from_pretrained(PRETRAINED_MODEL_NAME,
                                                          num_labels=2,
                                                          output_hidden_states=False,
                                                          output_attentions=False)

    if device.type != 'cpu':
      # Run model in cuda if available
      print("Running model in CUDA")
      model.cuda()

    optimizer = AdamW(model.parameters(),lr=lr, eps=eps, correct_bias=False)
    total_steps = len(train_data_loader) * EPOCHS

    # Create the learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, 
                                                num_warmup_steps=0,
                                                num_training_steps=total_steps)
    
    # We'll store a number of quantities such as training and validation loss, 
    # validation accuracy, and timings.
    training_stats = []

    # Creates a GradScaler once at the beginning of training.
    scaler = GradScaler()

    # Measure the total training time for the whole run.
    total_t0 = time.time()

    # For each epoch...
    for epoch_i in range(0, EPOCHS):
      print("")
      print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, EPOCHS))
      print('Training...')

      # Measure how long the training epoch takes.
      t0 = time.time()

      # reset the total loss for this training epoch
      total_train_loss = 0

      # Put the model in training mode so that dropout and batch norm layers can behave accordingly
      # https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
      model.train()

      # For each batch of training data...
      for step, batch in enumerate(train_data_loader):
        # Progress update every 40 batches.
        if step % 40 == 0 and not step == 0:
            # Calculate elapsed time in minutes.
            elapsed = format_time(time.time() - t0)
            
            # Report progress.
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_data_loader), elapsed))

        b_input_ids = batch['input_ids'].to(device)
        b_input_mask = batch['attention_mask'].to(device)
        b_labels = batch['targets'].to(device)

        # Clear previously calculated gradients before backward pass
        model.zero_grad()  

        # Perform forward pass. Here we also obtain loss since we passed labels
        # logits - model outputs prior to activation
        # Runs the forward pass with autocasting.
        with autocast():
            loss, logits = model(b_input_ids,
                              token_type_ids=None,
                              attention_mask=b_input_mask,
                              labels=b_labels)
        total_train_loss += loss.item()
        # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
        # Backward passes under autocast are not recommended.
        # Backward ops run in the same dtype autocast chose for corresponding forward ops.

        # Perform backward pass to calculate gradients
        scaler.scale(loss).backward()
        
        # scaler.step() first unscales the gradients of the optimizer's assigned params.
        # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
        # otherwise, optimizer.step() is skipped.
        scaler.step(optimizer)

        # Updates the scale for next iteration.
        scaler.update()

        # Clip gradients to 1.0 to avoid exploding gradients. Uncomment below line to observe performance change
        # torch.nn.utils.clip_grad_norm(model.parameters(), 1.0)
        # optimizer.step()

        # Update learning rate
        scheduler.step()

      # Average training loss over all batches in current epoch
      avg_train_loss = total_train_loss / len(train_data_loader)

      # Measure how long this epoch took
      training_time = format_time(time.time() - t0)
      print("")
      print("  Average training loss: {0:.2f}".format(avg_train_loss))
      print("  Training epcoh took: {:}".format(training_time))

      # After the completion of each training epoch, measure our performance on
      # our validation set.

      print("")
      print("Running Validation...")

      t0 = time.time()

      # Put the model in evaluation mode--the dropout layers behave differently
      # during evaluation.
      model.eval()

      # Tracking variables 
      total_eval_accuracy = 0
      total_eval_loss = 0
      nb_eval_steps = 0

      # Evaluate data for one epoch
      for batch in val_data_loader:
        b_input_ids = batch['input_ids'].to(device)
        b_input_mask = batch['attention_mask'].to(device)
        b_labels = batch['targets'].to(device)

        # Tell pytorch not to bother with constructing the compute graph during
        # the forward pass, since this is only needed for backprop (training).
        with torch.no_grad():    
          loss, logits = model(b_input_ids,
                              token_type_ids=None,
                              attention_mask=b_input_mask,
                              labels=b_labels)
          total_eval_loss += loss.item()

          # Move logits and labels to CPU
          logits = logits.detach().cpu().numpy()
          label_ids = b_labels.to('cpu').numpy()

          # Calculate the accuracy for this batch of test sentences, and
          # accumulate it over all batches.
          total_eval_accuracy += flat_accuracy(logits, label_ids)
        
        # Report final accuracy for this validation run
        avg_val_accuracy = total_eval_accuracy / len(val_data_loader)
        
    # Calculate the average loss over all of the batches.
    avg_val_loss = total_eval_loss / len(val_data_loader)

    # Measure how long the validation run took.
    validation_time = format_time(time.time() - t0)

    print("  Validation Loss: {0:.2f}".format(avg_val_loss))
    print("  Validation took: {:}".format(validation_time))

    # Record all statistics from this epoch.
    training_stats.append(
        {
            'epoch': epoch_i + 1,
            'Training Loss': avg_train_loss,
            'Valid. Loss': avg_val_loss,
            'Valid. Accur.': avg_val_accuracy,
            'Training Time': training_time,
            'Validation Time': validation_time
        }
    )

    print("")
    print("Training complete!")
    print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))


    # Save the model
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    print("saving model to {}".format(model_save_path))

    model_to_save = model.module if hasattr(model, 'module') else model # Take care of distributed/parallel training
    model_to_save.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)

if __name__ == '__main__':
    main()

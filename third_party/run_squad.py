# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors, 
# The HuggingFace Inc. team, and The XTREME Benchmark Authors.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Finetuning the library models for question-answering on SQuAD (DistilBERT, Bert, XLM, XLNet)."""


import argparse
import glob
import logging
import os
import random
import timeit

import re
import pickle

from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

import random

import torch.nn as nn

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

import copy
import datetime
import time


import sys
sys.path.append(r"/home/yehai/xtreme/")
from transformersFast.src.transformers import (
  WEIGHTS_NAME,
  AdamW,
  AlbertConfig,
  AlbertForQuestionAnswering,
  AlbertTokenizer,
  BertConfig,
  BertForQuestionAnswering,
  BertForMaskedLM,
  BertTokenizer,
  DistilBertConfig,
  DistilBertForQuestionAnswering,
  DistilBertTokenizer,
  XLMConfig,
  XLMForQuestionAnswering,
  XLMTokenizer,
  XLNetConfig,
  XLNetForQuestionAnswering,
  XLMRobertaForMaskedLM,
  XLNetTokenizer,
  get_linear_schedule_with_warmup,
  get_constant_schedule_with_warmup,
  XLMRobertaTokenizer
)

from transformers.data.metrics.squad_metrics import (
  compute_predictions_log_probs,
  compute_predictions_logits,
  squad_evaluate,
)

from transformersFast.src.transformers.modeling_xlm_roberta import XLMRobertaForQuestionAnswering, XLMRobertaConfig
#from transformersFast.src.transformers.modeling_roberta import RobertaClassificationHead



from processors.squad import (
  SquadResult,
  SquadV1Processor,
  SquadV2Processor,
  squad_convert_examples_to_features
)

try:
  from torch.utils.tensorboard import SummaryWriter
except ImportError:
  from tensorboardX import SummaryWriter


logger = logging.getLogger(__name__)

ALL_MODELS = sum(
  (tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig, XLNetConfig, XLMConfig)),
  (),
)

MODEL_CLASSES = {
  "bert": (BertConfig, BertForQuestionAnswering, BertTokenizer),
  "xlnet": (XLNetConfig, XLNetForQuestionAnswering, XLNetTokenizer),
  "xlm": (XLMConfig, XLMForQuestionAnswering, XLMTokenizer),
  "distilbert": (DistilBertConfig, DistilBertForQuestionAnswering, DistilBertTokenizer),
  "albert": (AlbertConfig, AlbertForQuestionAnswering, AlbertTokenizer),
  "xlm-roberta": (XLMRobertaConfig, XLMRobertaForQuestionAnswering, XLMRobertaTokenizer)
}

MODEL_MASK_CLASSES = {
  "bert": (BertConfig, BertForMaskedLM, BertTokenizer),
  "xlm-roberta": (XLMRobertaConfig, XLMRobertaForMaskedLM, XLMRobertaTokenizer),
}


def set_seed(args):
  random.seed(args.seed)
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  if args.n_gpu > 0:
    torch.cuda.manual_seed_all(args.seed)


def to_list(tensor):
  return tensor.detach().cpu().tolist()


def train(args, train_dataset, model, tokenizer):
  """ Train the model """
  if args.local_rank in [-1, 0]:
    tb_writer = SummaryWriter()

  args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
  train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
  train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

  if args.max_steps > 0:
    t_total = args.max_steps
    args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
  else:
    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

  # Prepare optimizer and schedule (linear warmup and decay)
  no_decay = ["bias", "LayerNorm.weight"]
  optimizer_grouped_parameters = [
    {
      "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
      "weight_decay": args.weight_decay,
    },
    {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
  ]
  optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
  scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
  )

  # Check if saved optimizer or scheduler states exist
  if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
    os.path.join(args.model_name_or_path, "scheduler.pt")
  ):
    # Load in optimizer and scheduler states
    optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
    scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

  if args.fp16:
    try:
      from apex import amp
    except ImportError:
      raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

    model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

  # multi-gpu training (should be after apex fp16 initialization)
  if args.n_gpu > 1:
    model = torch.nn.DataParallel(model)

  # Distributed training (should be after apex fp16 initialization)
  if args.local_rank != -1:
    model = torch.nn.parallel.DistributedDataParallel(
      model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
    )

  # Train!
  logger.info("***** Running training *****")
  logger.info("  Num examples = %d", len(train_dataset))
  logger.info("  Num Epochs = %d", args.num_train_epochs)
  logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
  logger.info(
    "  Total train batch size (w. parallel, distributed & accumulation) = %d",
    args.train_batch_size
    * args.gradient_accumulation_steps
    * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
  )
  logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
  logger.info("  Total optimization steps = %d", t_total)

  global_step = 1
  epochs_trained = 0
  steps_trained_in_current_epoch = 0
  # Check if continuing training from a checkpoint
  if os.path.exists(args.model_name_or_path):
    try:
      # set global_step to gobal_step of last saved checkpoint from model path
      checkpoint_suffix = args.model_name_or_path.split("-")[-1].split("/")[0]
      global_step = int(checkpoint_suffix)
      epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
      steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

      logger.info("  Continuing training from checkpoint, will skip to saved global_step")
      logger.info("  Continuing training from epoch %d", epochs_trained)
      logger.info("  Continuing training from global step %d", global_step)
      logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
    except ValueError:
      logger.info("  Starting fine-tuning.")

  tr_loss, logging_loss = 0.0, 0.0
  model.zero_grad()
  train_iterator = trange(
    epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
  )
  # Added here for reproductibility
  set_seed(args)

  for _ in train_iterator:
    epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
    for step, batch in enumerate(epoch_iterator):

      # Skip past any already trained steps if resuming training
      if steps_trained_in_current_epoch > 0:
        steps_trained_in_current_epoch -= 1
        continue

      model.train()
      batch = tuple(t.to(args.device) for t in batch)

      inputs = {
        "input_ids": batch[0],
        "attention_mask": batch[1],
        "token_type_ids": None if args.model_type in ["xlm", "xlm-roberta", "distilbert"] else batch[2],
        "start_positions": batch[3],
        "end_positions": batch[4],
      }

      if args.model_type in ["xlnet", "xlm"]:
        inputs.update({"cls_index": batch[5], "p_mask": batch[6]})
        if args.version_2_with_negative:
          inputs.update({"is_impossible": batch[7]})
      if args.model_type == "xlm":
        inputs["langs"] = batch[7]
      outputs = model(**inputs)
      # model outputs are always tuple in transformers (see doc)
      loss = outputs[0]

      if args.n_gpu > 1:
        loss = loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training
      if args.gradient_accumulation_steps > 1:
        loss = loss / args.gradient_accumulation_steps

      if args.fp16:
        with amp.scale_loss(loss, optimizer) as scaled_loss:
          scaled_loss.backward()
      else:
        loss.backward()

      tr_loss += loss.item()
      if (step + 1) % args.gradient_accumulation_steps == 0:
        if args.fp16:
          torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
        else:
          torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        optimizer.step()
        scheduler.step()  # Update learning rate schedule
        model.zero_grad()
        global_step += 1

        # Log metrics
        if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
          # Only evaluate when single GPU otherwise metrics may not average well
          if args.local_rank == -1 and args.evaluate_during_training:
            results = evaluate(args, model, tokenizer)
            for key, value in results.items():
              tb_writer.add_scalar("eval_{}".format(key), value, global_step)
          tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
          tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
          logging_loss = tr_loss

        # Save model checkpoint
        if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
          output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
          if not os.path.exists(output_dir):
            os.makedirs(output_dir)
          # Take care of distributed/parallel training
          model_to_save = model.module if hasattr(model, "module") else model
          model_to_save.save_pretrained(output_dir)
          tokenizer.save_pretrained(output_dir)

          torch.save(args, os.path.join(output_dir, "training_args.bin"))
          logger.info("Saving model checkpoint to %s", output_dir)

          torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
          torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
          logger.info("Saving optimizer and scheduler states to %s", output_dir)

      if args.max_steps > 0 and global_step > args.max_steps:
        epoch_iterator.close()
        break
    if args.max_steps > 0 and global_step > args.max_steps:
      train_iterator.close()
      break

  if args.local_rank in [-1, 0]:
    tb_writer.close()

  return global_step, tr_loss / global_step



## ----> 
def HIL_multi_arm_ucb_self_train(args, tokenizer, config, lang2id):  
  """ Train the model """
  ##### Dont change the code [Start]
  np.random.seed(args.seed)
  train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False, output_examples=False, language=args.eval_lang, lang2id=lang2id)

  train_set_index = list(range(len(train_dataset)))
  random.shuffle(train_set_index)

  train_set = []
  for ii in train_set_index[:args.training_data_num]:
    train_set.append(train_dataset[ii] + (list(np.random.binomial(size=1, n=1, p = args.leave_feedback_rate ))[0], ) ) ## add the mask for leaving human feedback or nots
  ##### Dont change the code [End]
  
  
  
  args.train_batch_size = args.per_gpu_train_batch_size
  train_sampler = SequentialSampler(train_set) 
  train_dataloader = DataLoader(train_set, sampler=train_sampler, batch_size=args.train_batch_size)

  
  ## load the models
  assert len(args.source_model_path) == args.num_of_experts
  models = []
  for path_ in args.source_model_path:
    args.model_type = args.model_type.lower()
    _, model_class, _ = MODEL_CLASSES[args.model_type]
    models.append(
      model_class.from_pretrained(
      path_,
      from_tf=bool(".ckpt" in path_),
      config=config,
      cache_dir=args.cache_dir if args.cache_dir else None,
      )
    )
  models = torch.nn.ModuleList(models).to(args.device)
  

      

  """
  roberta.encoder.layer.2.attention.self.query.weight | 0
  roberta.encoder.layer.2.attention.self.query.bias | 0
  roberta.encoder.layer.2.attention.self.key.weight | 0
  roberta.encoder.layer.2.attention.self.key.bias | 0
  roberta.encoder.layer.2.attention.self.value.weight | 0
  roberta.encoder.layer.2.attention.self.value.bias | 0
  roberta.encoder.layer.2.attention.output.dense.weight | 0
  roberta.encoder.layer.2.attention.output.dense.bias | 0
  roberta.encoder.layer.2.attention.output.LayerNorm.weight | 0
  roberta.encoder.layer.2.attention.output.LayerNorm.bias | 0
  roberta.encoder.layer.2.intermediate.dense.weight | 0
  roberta.encoder.layer.2.intermediate.dense.bias | 0
  roberta.encoder.layer.2.output.dense.weight | 0
  roberta.encoder.layer.2.output.dense.bias | 0
  roberta.encoder.layer.2.output.LayerNorm.weight | 0
  roberta.encoder.layer.2.output.LayerNorm.bias | 0
  """


  for name, param in models.named_parameters():
    if '.encoder.layer' in name:
      layer_id = int(re.findall(r"\d+", name)[1])
      if not (12 - layer_id <= args.num_layer_to_update):
        param.requires_grad = False
    elif ('.embeddings' in name ) :
      param.requires_grad = False

    print ("%s | %d" % (name, param.requires_grad) )
  
  # Prepare optimizer and schedule (linear warmup and decay)
  no_decay = ["bias", "LayerNorm.weight"]
  optimizer_grouped_parameters = [
    {
      "params": [p for n, p in models.named_parameters() if not any(nd in n for nd in no_decay)],
      "weight_decay": args.weight_decay,
    },
    {"params": [p for n, p in models.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
  ]





  optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
  scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=0)




  # Train!
  logger.info("***** Running training *****")
  logger.info("  Num examples = %d", len(train_set))
  logger.info("  Num Epochs = %d", args.num_train_epochs)
  logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)



  global_step = 1
  epochs_trained = 0


  models.zero_grad()


  train_iterator = trange(
    epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
  )
  # Added here for reproductibility
  set_seed(args)

  all_data_num = 0.
  num_of_model_annotated=0.
  num_annotated = 0.
  num_annotated_all = []


  f1_test = []
  em_test = []


  program_running_time_seconds = 0.


  
  """
  T = 1000 
  N = 10 

  true_rewards = np.random.uniform(low=0, high=1, size=N) 
  estimated_rewards = np.zeros(N) 
  chosen_count = np.zeros(N) 
  total_reward = 0 

  def calculate_delta(T, item):
    if chosen_count[item] == 0:
        return 1
    else:
        return np.sqrt(2 * np.log(T) / chosen_count[item])

  def UCB(t, N):
    upper_bound_probs = [estimated_rewards[item] + calculate_delta(t, item) for item in range(N)]
    item = np.argmax(upper_bound_probs)
    reward = np.random.binomial(n=1, p=true_rewards[item])
    return item, reward

  for t in range(1, T): 
   item, reward = UCB(t, N)
   total_reward += reward 
   estimated_rewards[item] = ((t - 1) * estimated_rewards[item] + reward) / t
   chosen_count[item] += 1
  """

  ### [START] UCB [START]
  total_reward = 0 
  estimated_rewards = np.zeros(args.num_of_experts) 
  chosen_count = np.zeros(args.num_of_experts) 
  total_steps = 0.

  def calculate_delta(model_id):
    if chosen_count[model_id] == 0:
        return 1.
    else:
        return np.sqrt(2 * np.log(total_steps) / chosen_count[model_id])

  def ucb_action(model_num):
    if total_steps == 0:
      random_model_ids = list(range(args.num_of_experts))
      random.shuffle(random_model_ids)
      return random_model_ids

    upper_bound_probs = [estimated_rewards[model_id] + calculate_delta(model_id) for model_id in range(model_num)]
    #upper_bound_probs = estimated_rewards ## dont consider the exploration
    model_id = np.argsort(upper_bound_probs)  ## last two model ids are needed
    return model_id
  ### [END] UCB [START]


  ### 8888
  result_before_adaptation_em = []
  result_before_adaptation_f1 = []
  if args.eval_before_adapt:
    for model_id, model in enumerate(models):
      result = evaluate(args, model, tokenizer, language=args.eval_lang)
      logger.info("***** [Before Adaptation] [Model %d] Current Eval results on Target*****", model_id)
      for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(result[key]))
      result_before_adaptation_em.append(str(result['exact']))
      result_before_adaptation_f1.append(str(result['f1']))


  for _ in train_iterator:
    epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
    for step, batch in enumerate(epoch_iterator):      
      batch = tuple(t.to(args.device) for t in batch)
      inputs = {
        "input_ids": batch[0],
        "attention_mask": batch[1],
        "token_type_ids": None if args.model_type in ["xlm", "xlm-roberta", "distilbert"] else batch[2],
        "start_positions": batch[3],
        "end_positions": batch[4],
      }

      if args.model_type in ["xlnet", "xlm"]:
        inputs.update({"cls_index": batch[5], "p_mask": batch[6]})
        if args.version_2_with_negative:
          inputs.update({"is_impossible": batch[7]})
      if args.model_type == "xlm":
        inputs["langs"] = batch[7]

      ## choose two models using ucb
      ranked_model_ids = ucb_action(args.num_of_experts)
      model_id_0 = ranked_model_ids[-1]  ### the best model is the last one
      model = models[model_id_0]
      # 1. model predicts
      models.train()

      outputs = model(**inputs)
      pred_start_logit = outputs[1]
      pred_end_logit = outputs[2]
      pred_start_position = torch.argmax(pred_start_logit, dim = 1)
      pred_end_position = torch.argmax(pred_end_logit, dim = 1)


      
      # 2. obtain the user feedback
      ## all data num
      _, loss_hard_start_, loss_hard_end_ = qa_loss(torch.softmax(pred_start_logit, dim = 1).detach(), torch.softmax(pred_end_logit, dim = 1).detach(), copy.deepcopy(pred_start_position), copy.deepcopy(pred_end_position), return_start_end=True)
      
      all_data_num += loss_hard_start_.size(0) + loss_hard_end_.size(0)

      ## [Model] mask for model certain data
      reward_mask_pos_model_start = (loss_hard_start_ < args.threshold).view(-1)  ## [model]
      reward_mask_pos_model_end = (loss_hard_end_ < args.threshold).view(-1)  ## [model]



      
      ## [Human] feedback
      ave_pred_start_logit = pred_start_logit 
      ave_pred_end_logit = pred_end_logit 

      #assert args.topk == 1
      if args.topk != -1:
        topk_selection_start = torch.topk(ave_pred_start_logit, int(args.topk), dim = 1)[1] == batch[3].view(-1,1)
        topk_selection_end = torch.topk(ave_pred_end_logit, int(args.topk), dim = 1)[1] == batch[4].view(-1,1)
      else:
        topk_selection_start = torch.topk(ave_pred_start_logit, ave_pred_start_logit.size(1), dim = 1)[1] == batch[3].view(-1,1)
        topk_selection_end = torch.topk(ave_pred_end_logit, ave_pred_end_logit.size(1), dim = 1)[1] == batch[4].view(-1,1)

      reward_mask_pos_after_topk_selection_start = (torch.sum(topk_selection_start, dim = 1).view(-1, 1) == 1).view(-1)   ## [human]
      reward_mask_pos_after_topk_selection_end = (torch.sum(topk_selection_end, dim = 1).view(-1, 1) == 1).view(-1)     ## [human]

      ## [Model] and [Human] 
      ### remove the data that has no human feedback
      mask_has_human_feedback = batch[-1].view(-1) == 1
      num_annotated += torch.sum(mask_has_human_feedback) * 2


      ## [Model] * mask_has_human_feedback
      reward_mask_pos_model_start = reward_mask_pos_model_start * ~mask_has_human_feedback
      reward_mask_pos_model_end = reward_mask_pos_model_end * ~mask_has_human_feedback


      num_of_model_annotated += torch.mean(reward_mask_pos_model_start*1.) + torch.mean(reward_mask_pos_model_end*1.)


      ## [Human] * mask_has_human_feedback
      reward_mask_pos_after_topk_selection_start = reward_mask_pos_after_topk_selection_start * mask_has_human_feedback
      reward_mask_pos_after_topk_selection_end = reward_mask_pos_after_topk_selection_end * mask_has_human_feedback

      ## [Human] start and end should both be 1
      true_reward = reward_mask_pos_after_topk_selection_start * reward_mask_pos_after_topk_selection_end
      reward_mask_pos_after_topk_selection_start = true_reward
      reward_mask_pos_after_topk_selection_end = true_reward


      ## [Human] either start and end should be 1
      #true_reward = reward_mask_pos_after_topk_selection_start + reward_mask_pos_after_topk_selection_end
      #reward_mask_pos_after_topk_selection_start = true_reward
      #reward_mask_pos_after_topk_selection_end = true_reward
      #print (true_reward)



      ## obtain the predicts
      pred_start_position_final = ~reward_mask_pos_after_topk_selection_start * pred_start_position + reward_mask_pos_after_topk_selection_start * copy.deepcopy(batch[3])
      pred_end_position_final = ~reward_mask_pos_after_topk_selection_end * pred_end_position + reward_mask_pos_after_topk_selection_end * copy.deepcopy(batch[4])



      ## obtain the rewards
      rewards_model_start = torch.tensor([0.] * pred_start_logit.size(0)).to(args.device)
      rewards_model_end = torch.tensor([0.] * pred_end_logit.size(0)).to(args.device)
      

      assert args.neg_reward == 0
      rewards_model_start[reward_mask_pos_after_topk_selection_start] = 1
      rewards_model_end[reward_mask_pos_after_topk_selection_end] = 1

      

      if args.training_mode == 'semi':
        rewards_model_start[reward_mask_pos_model_start] = 1
        rewards_model_end[reward_mask_pos_model_end] = 1



      ### mode 2
      cur_reward_model = (torch.sum(reward_mask_pos_after_topk_selection_start*1.) + torch.sum(reward_mask_pos_after_topk_selection_end*1.)) / 2.
      cur_reward_model = cur_reward_model.cpu().numpy()

      estimated_rewards[model_id_0] = ( chosen_count[model_id_0] * estimated_rewards[model_id_0] + cur_reward_model ) / (chosen_count[model_id_0] + torch.sum(mask_has_human_feedback) + 1e-10)
      chosen_count[model_id_0] += torch.sum(mask_has_human_feedback).cpu().numpy()      
      total_steps += torch.sum(mask_has_human_feedback).cpu().numpy()                   



      
      # 3. model updates 
      start_time = time.time() 

      _, loss_hard_start_, loss_hard_end_ = qa_loss(pred_start_logit, pred_end_logit, copy.deepcopy(pred_start_position_final), copy.deepcopy(pred_end_position_final), return_start_end=True) # (batch,)    
      loss = torch.sum(rewards_model_start * loss_hard_start_) / (torch.sum(rewards_model_start > 0) + 1e-10) +\
                  torch.sum(rewards_model_end * loss_hard_end_) / (torch.sum(rewards_model_end > 0) + 1e-10)
      loss /= 2

      

      loss.backward()
      torch.nn.utils.clip_grad_norm_(models.parameters(), args.max_grad_norm)
      optimizer.step()
      scheduler.step()  # Update learning rate schedule
      models.zero_grad()
      end_time = time.time() #datetime.datetime.now()
      program_running_time_seconds += end_time - start_time #(end_time - start_time).seconds
      
      global_step += 1

      if args.logging_steps > 0 and global_step % args.logging_steps == 0:
        ## model evaluation
        num_annotated_all.append(str(num_annotated))
        ## select a model for evaluation
        ranked_model_ids = ucb_action(args.num_of_experts)
        selected_model_id = ranked_model_ids[-1]  ### the best model is the last one

        result = evaluate(args, models[selected_model_id], tokenizer, language=args.eval_lang)
        f1_test.append(str(result['f1']))
        em_test.append(str(result['exact']))
        result["language"] = args.eval_lang
        logger.info("***** [Model %d ] Current Eval results *****", selected_model_id)
        for key in sorted(result.keys()):
          logger.info("  %s = %s", key, str(result[key]))

          
        logger.info("pro. of annotated: %lf", num_annotated / float(all_data_num) )
        logger.info("all_data_num: %d | num_annotated: %d | num_of_model_annotated: %d", all_data_num, num_annotated.cpu().numpy(), num_of_model_annotated.cpu().numpy())
        logger.info("estimated_rewards: %s", ' '.join([str(q) for q in list(estimated_rewards)]))
        logger.info("chosen_count: %s", ' '.join([str(q) for q in list(chosen_count)]))


        
        
  ## model evaluation
  num_annotated_all.append(str(num_annotated))
  ## select a model for evaluation
  ranked_model_ids = ucb_action(args.num_of_experts)
  selected_model_id = ranked_model_ids[-1]
  result = evaluate(args, models[selected_model_id], tokenizer, language=args.eval_lang)
  f1_test.append(str(result['f1']))
  em_test.append(str(result['exact']))
  result["language"] = args.eval_lang
  logger.info("***** [Model %d ] Current Eval results *****", selected_model_id)
  for key in sorted(result.keys()):
    logger.info("  %s = %s", key, str(result[key]))
        
  logger.info("pro. of annotated: %lf", num_annotated / float(all_data_num) )
  logger.info("all_data_num: %d | num_annotated: %d | num_of_model_annotated: %d", all_data_num, num_annotated, num_of_model_annotated.cpu().numpy())
  logger.info("estimated_rewards: %s", ' '.join([str(q) for q in list(estimated_rewards)]))
  logger.info("chosen_count: %s", ' '.join([str(q) for q in list(chosen_count)]))


  print ("estimated_rewards: %s" % (' '.join([str(q) for q in list(estimated_rewards)])) )
  print ("chosen_count: %s" % ( ' '.join([str(q) for q in list(chosen_count)]) ))
  print ("pro. of annotated: %lf" % ( num_annotated / float(all_data_num)))
  print ('all_data_num: %d' % (all_data_num)  )
  print ('num_annotated_all:')
  print ("result before adaptation f1:")
  print ('\n'.join(result_before_adaptation_f1))
  print ("result before adaptation em:")
  print ('\n'.join(result_before_adaptation_em))
  print ('\n\n')

  print ('\n'.join(num_annotated_all))
  print ('\n\n')
  print ('f1_test:')
  print ('\n'.join(f1_test))
  print ('\n\n')
  print ('em_test:')
  print ('\n'.join(em_test))
  print ('\n\n')
  

  print ('---> time used is:')
  print ('%lf'  % (program_running_time_seconds) )

  return None #global_step, tr_loss / global_step


## --->
def HIL_multi_arm_ucb_co_train_preference_f1(args, tokenizer, config, lang2id):  
  """ Train the model """
  ##### Dont change the code [Start]
  np.random.seed(args.seed)
  train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False, output_examples=False, language=args.eval_lang, lang2id=lang2id)

  train_set_index = list(range(len(train_dataset)))
  random.shuffle(train_set_index)

  train_set = []
  for ii in train_set_index[:args.training_data_num]:
    train_set.append(train_dataset[ii] + (list(np.random.binomial(size=1, n=1, p = args.leave_feedback_rate ))[0], ) ) ## add the mask for leaving human feedback or nots
  ##### Dont change the code [End]
  
  
  
  args.train_batch_size = args.per_gpu_train_batch_size
  train_sampler = SequentialSampler(train_set) 
  train_dataloader = DataLoader(train_set, sampler=train_sampler, batch_size=args.train_batch_size)

  
  ## load the models
  assert len(args.source_model_path) == args.num_of_experts
  models = []
  for path_ in args.source_model_path:
    args.model_type = args.model_type.lower()
    _, model_class, _ = MODEL_CLASSES[args.model_type]
    models.append(
      model_class.from_pretrained(
      path_,
      from_tf=bool(".ckpt" in path_),
      config=config,
      cache_dir=args.cache_dir if args.cache_dir else None,
      )
    )
  models = torch.nn.ModuleList(models).to(args.device)
  
  #back_models = copy.deepcopy(models)
  #back_models_2 = copy.deepcopy(models[:1])

      

  """
  roberta.encoder.layer.2.attention.self.query.weight | 0
  roberta.encoder.layer.2.attention.self.query.bias | 0
  roberta.encoder.layer.2.attention.self.key.weight | 0
  roberta.encoder.layer.2.attention.self.key.bias | 0
  roberta.encoder.layer.2.attention.self.value.weight | 0
  roberta.encoder.layer.2.attention.self.value.bias | 0
  roberta.encoder.layer.2.attention.output.dense.weight | 0
  roberta.encoder.layer.2.attention.output.dense.bias | 0
  roberta.encoder.layer.2.attention.output.LayerNorm.weight | 0
  roberta.encoder.layer.2.attention.output.LayerNorm.bias | 0
  roberta.encoder.layer.2.intermediate.dense.weight | 0
  roberta.encoder.layer.2.intermediate.dense.bias | 0
  roberta.encoder.layer.2.output.dense.weight | 0
  roberta.encoder.layer.2.output.dense.bias | 0
  roberta.encoder.layer.2.output.LayerNorm.weight | 0
  roberta.encoder.layer.2.output.LayerNorm.bias | 0
  3.roberta.pooler.dense.weight | 1
  3.roberta.pooler.dense.bias | 1
  3.qa_outputs.weight | 1
  3.qa_outputs.bias | 1
  """

  for name, param in models.named_parameters():

    if 'roberta.encoder.layer' in name:
      layer_id = int(re.findall(r"\d+", name)[1])
      if not (12 - layer_id <= args.num_layer_to_update):
        param.requires_grad = False
    elif ('roberta.embeddings' in name ) :
      param.requires_grad = False

    
    #if ('roberta.embeddings' in name ) or ('intermediate.dense' in name) or ('output.dense' in name) or ('output.LayerNorm' in name) or ('qa_outputs' in name) or ('pooler.dense' in name):
    #if ('roberta.embeddings' in name ) or ('attention' in name) or ('qa_outputs' in name) or ('pooler.dense' in name):
    #  param.requires_grad = False

    print ("%s | %d" % (name, param.requires_grad) )
  
  # Prepare optimizer and schedule (linear warmup and decay)
  no_decay = ["bias", "LayerNorm.weight"]
  optimizer_grouped_parameters = [
    {
      "params": [p for n, p in models.named_parameters() if not any(nd in n for nd in no_decay)],
      "weight_decay": args.weight_decay,
    },
    {"params": [p for n, p in models.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
  ]
  optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
  scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=0)




  # Train!
  logger.info("***** Running training *****")
  logger.info("  Num examples = %d", len(train_set))
  logger.info("  Num Epochs = %d", args.num_train_epochs)
  logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)



  global_step = 1
  epochs_trained = 0


  models.zero_grad()


  train_iterator = trange(
    epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
  )
  # Added here for reproductibility
  set_seed(args)

  all_data_num = 0.
  num_of_model_annotated=0.
  num_annotated = 0.
  num_annotated_all = []
  num_of_model_same = 0.


  f1_test = []
  em_test = []


  program_running_time_seconds = 0.


  
  """
  T = 1000 
  N = 10 

  true_rewards = np.random.uniform(low=0, high=1, size=N) 
  estimated_rewards = np.zeros(N) 
  chosen_count = np.zeros(N) 
  total_reward = 0 

  def calculate_delta(T, item):
    if chosen_count[item] == 0:
        return 1
    else:
        return np.sqrt(2 * np.log(T) / chosen_count[item])

  def UCB(t, N):
    upper_bound_probs = [estimated_rewards[item] + calculate_delta(t, item) for item in range(N)]
    item = np.argmax(upper_bound_probs)
    reward = np.random.binomial(n=1, p=true_rewards[item])
    return item, reward

  for t in range(1, T): 
   item, reward = UCB(t, N)
   total_reward += reward 
   estimated_rewards[item] = ((t - 1) * estimated_rewards[item] + reward) / t
   chosen_count[item] += 1
  """

  ### [START] UCB ###
  total_reward = 0 
  estimated_rewards = np.zeros(args.num_of_experts) 
  chosen_count = np.zeros(args.num_of_experts) 
  total_steps = 0.

  def calculate_delta(model_id):
    if chosen_count[model_id] == 0:
        return 1.
    else:
        return np.sqrt(2 * np.log(total_steps) / chosen_count[model_id])

  def ucb_action(model_num):
    upper_bound_probs = [estimated_rewards[model_id] + calculate_delta(model_id) for model_id in range(model_num)]
    model_id = np.argsort(upper_bound_probs)  ## last two model ids are needed
    return model_id



  from itertools import combinations
  combins = [c for c in  combinations(range(args.num_of_experts), 2)]
  random.shuffle(combins)
  co_estimated_rewards = np.zeros(len(combins)) 
  co_chosen_count = np.zeros(len(combins)) 
  co_total_steps = 0.
  
  def calculate_delta_co(model_id):
    if co_chosen_count[model_id] == 0:
        return 1.
    else:
        return np.sqrt(2 * np.log(co_total_steps) / co_chosen_count[model_id])

  def ucb_action_co():
    ## (a + b ) / 2
    if args.co_ucb == 1:
      upper_bound_probs = [ (estimated_rewards[combins[model_id][0]] + estimated_rewards[combins[model_id][1]] ) / 2 + calculate_delta_co(model_id) for model_id in range(len(combins)) ]
    elif args.co_ucb == 0:
      upper_bound_probs = [ co_estimated_rewards[model_id] + calculate_delta_co(model_id) for model_id in range(len(combins)) ]
    elif args.co_ucb == 2:
      upper_bound_probs = [  co_estimated_rewards[model_id] + (estimated_rewards[combins[model_id][0]] + estimated_rewards[combins[model_id][1]] ) / 2 + 2 * calculate_delta_co(model_id) for model_id in range(len(combins)) ]
    
    
    model_id = np.argmax(upper_bound_probs)  ## the last one has to be combins[model_id]
    model_id_0, model_id_1 = combins[model_id]
    


    return model_id_0, model_id_1, model_id

  ### [END] UCB ###


  ##8888
  result_before_adaptation_em = []
  result_before_adaptation_f1 = []
  ### Evaluation before adaptation
  if args.eval_before_adapt == 1:
    for model_id, model in enumerate(models):
      result = evaluate(args, model, tokenizer, language=args.eval_lang)
      logger.info("***** [Before Adaptation] [Model %d] Current Eval results on Target*****", model_id)
      for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(result[key]))
      result_before_adaptation_em.append(str(result['exact']))
      result_before_adaptation_f1.append(str(result['f1']))
      
      






  for _ in train_iterator:
    epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
    for step, batch in enumerate(epoch_iterator):      
      batch = tuple(t.to(args.device) for t in batch)
      inputs = {
        "input_ids": batch[0],
        "attention_mask": batch[1],
        "token_type_ids": None if args.model_type in ["xlm", "xlm-roberta", "distilbert"] else batch[2],
        "start_positions": batch[3],
        "end_positions": batch[4],
      }

      if args.model_type in ["xlnet", "xlm"]:
        inputs.update({"cls_index": batch[5], "p_mask": batch[6]})
        if args.version_2_with_negative:
          inputs.update({"is_impossible": batch[7]})
      if args.model_type == "xlm":
        inputs["langs"] = batch[7]

      ## choose two models using co ucb
      model_id_0, model_id_1, combins_id = ucb_action_co()
      model, large_model = models[model_id_0], models[model_id_1]
      # 1. model predicts
      models.train()

      outputs = model(**inputs)
      outputs_large = large_model(**inputs)
      pred_start_logit = outputs[1]
      pred_end_logit = outputs[2]
      pred_start_position = torch.argmax(pred_start_logit, dim = 1)
      pred_end_position = torch.argmax(pred_end_logit, dim = 1)

      pred_start_logit_large = outputs_large[1]
      pred_end_logit_large = outputs_large[2]
      pred_start_position_large = torch.argmax(pred_start_logit_large, dim = 1)
      pred_end_position_large = torch.argmax(pred_end_logit_large, dim = 1)


      # 2. obtain the user feedback
      ## all data num
      all_data_num += pred_start_logit.size(0) + pred_start_logit.size(0)


      ## [Human] preference

      ### type 1      
      A_B_0 = 1 + torch.min( torch.cat( (batch[4].view(-1,1), pred_end_position.view(-1,1) ), dim = 1 ), dim = 1)[0] - torch.max( torch.cat( (batch[3].view(-1,1), pred_start_position.view(-1,1) ), dim = 1 ), dim = 1)[0] 
      A_B_0 = A_B_0.view(-1, 1)
      A_B_0 = A_B_0 * (A_B_0 >= 0)
      p_0 = A_B_0 / (1 + pred_end_position.view(-1, 1) - pred_start_position.view(-1, 1) + 1e-10)
      r_0 = A_B_0 / (1 + batch[4].view(-1, 1) - batch[3].view(-1, 1) + 1e-10)
      f_0 = 2 * p_0 * r_0 / (p_0 + r_0 + 1e-10)
      
      
      A_B_1 = 1 + torch.min( torch.cat( (batch[4].view(-1,1), pred_end_position_large.view(-1,1) ), dim = 1 ), dim = 1)[0] - torch.max( torch.cat( (batch[3].view(-1,1), pred_start_position_large.view(-1,1) ), dim = 1 ), dim = 1)[0] 
      A_B_1 = A_B_1.view(-1, 1)
      A_B_1 = A_B_1 * (A_B_1 >= 0)
      p_1 = A_B_1 / (1 + pred_end_position_large.view(-1, 1) - pred_start_position_large.view(-1, 1) + 1e-10)
      r_1 = A_B_1 / (1 + batch[4].view(-1, 1) - batch[3].view(-1, 1) + 1e-10)
      f_1 = 2 * p_1 * r_1 / (p_1 + r_1 + 1e-10)
      
      reward_model_0 = f_0.view(-1)
      #print (reward_model_0)
      reward_model_1 = f_1.view(-1)


      preference_model_0 = reward_model_0 > reward_model_1
      preference_model_1 = reward_model_1 > reward_model_0


      ### remove the data that has no human feedback
      mask_has_human_feedback = batch[-1].view(-1) == 1
      num_annotated += torch.sum(mask_has_human_feedback).cpu().numpy() * 2


      ## UCB updates
      reward_for_ucb = (torch.sum(preference_model_0) + torch.sum(preference_model_1) ) / 2
      reward_for_ucb = reward_for_ucb.cpu().numpy()


      ### update model 0
      estimated_rewards[model_id_0] = ( chosen_count[model_id_0] * estimated_rewards[model_id_0] + torch.sum(preference_model_0).cpu().numpy() ) / (chosen_count[model_id_0] + torch.sum(mask_has_human_feedback) + 1e-10)
      chosen_count[model_id_0] += torch.sum(mask_has_human_feedback).cpu().numpy()
      ### update model 1
      estimated_rewards[model_id_1] = ( chosen_count[model_id_1] * estimated_rewards[model_id_1] + torch.sum(preference_model_1).cpu().numpy() ) / (chosen_count[model_id_1] + torch.sum(mask_has_human_feedback) + 1e-10)
      chosen_count[model_id_1] += torch.sum(mask_has_human_feedback).cpu().numpy()

      total_steps += torch.sum(mask_has_human_feedback).cpu().numpy() * 2

      ### update model_0,1
      co_estimated_rewards[combins_id] =  ( co_chosen_count[combins_id] * co_estimated_rewards[combins_id] + reward_for_ucb ) / (co_chosen_count[combins_id] +  torch.sum(mask_has_human_feedback) + 1e-10)
      co_chosen_count[combins_id] += torch.sum(mask_has_human_feedback).cpu().numpy()
      co_total_steps += torch.sum(mask_has_human_feedback).cpu().numpy()



      
      # 3. model updates 
      start_time = time.time() #datetime.datetime.now()

      """
      _, loss_hard_start_, loss_hard_end_ = qa_loss(pred_start_logit, pred_end_logit, copy.deepcopy(pred_start_position), copy.deepcopy(pred_end_position), return_start_end=True) # (batch,)
      _, loss_hard_start_large_, loss_hard_end_large_ = qa_loss(pred_start_logit_large, pred_end_logit_large, copy.deepcopy(pred_start_position_large), copy.deepcopy(pred_end_position_large), return_start_end=True) # (batch,)

      loss = torch.sum(preference_model_0 * loss_hard_start_) / (torch.sum(preference_model_0 > 0) + 1e-10) +\
                  torch.sum(preference_model_0 * loss_hard_end_) / (torch.sum(preference_model_0 > 0) + 1e-10)
      loss += torch.sum(preference_model_1 * loss_hard_start_large_) / (torch.sum(preference_model_1 > 0) + 1e-10) +\
                  torch.sum(preference_model_1 * loss_hard_end_large_) / (torch.sum(preference_model_1 > 0) + 1e-10)
      loss /= 2
      """


      ## combine the benefits of the two models to update
      #preference_model = (reward_model_0 >= reward_model_1) * ~((reward_model_0 == 0) * (reward_model_1 == 0)) + (reward_model_1 > reward_model_0) * ~((reward_model_0 == 0) * (reward_model_1 == 0))
      #pred_start_combine = (reward_model_0 >= reward_model_1) * ~((reward_model_0 == 0) * (reward_model_1 == 0)) * pred_start_position + (reward_model_1 > reward_model_0) * ~((reward_model_0 == 0) * (reward_model_1 == 0)) * pred_start_position_large
      #pred_end_combine = (reward_model_0 >= reward_model_1) * ~((reward_model_0 == 0) * (reward_model_1 == 0)) * pred_end_position + (reward_model_1 > reward_model_0) * ~((reward_model_0 == 0) * (reward_model_1 == 0)) * pred_end_position_large

      #preference_model = (reward_model_0 > reward_model_1) + (reward_model_1 > reward_model_0)
      #pred_start_combine = (reward_model_0 > reward_model_1) * pred_start_position + (reward_model_1 > reward_model_0) * pred_start_position_large
      #pred_end_combine = (reward_model_0 > reward_model_1) * pred_end_position + (reward_model_1 > reward_model_0) * pred_end_position_large

      preference_model = preference_model_0 + preference_model_1
      pred_start_combine = preference_model_0 * pred_start_position + preference_model_1 * pred_start_position_large
      pred_end_combine = preference_model_0 * pred_end_position + preference_model_1 * pred_end_position_large



      _, loss_hard_start_, loss_hard_end_ = qa_loss(pred_start_logit, pred_end_logit, copy.deepcopy(pred_start_combine), copy.deepcopy(pred_end_combine), return_start_end=True) # (batch,)
      _, loss_hard_start_large_, loss_hard_end_large_ = qa_loss(pred_start_logit_large, pred_end_logit_large, copy.deepcopy(pred_start_combine), copy.deepcopy(pred_end_combine), return_start_end=True) # (batch,)

      loss = torch.sum(preference_model * loss_hard_start_) / (torch.sum(preference_model > 0) + 1e-10) +\
                  torch.sum(preference_model * loss_hard_end_) / (torch.sum(preference_model > 0) + 1e-10)
      loss += torch.sum(preference_model * loss_hard_start_large_) / (torch.sum(preference_model > 0) + 1e-10) +\
                  torch.sum(preference_model * loss_hard_end_large_) / (torch.sum(preference_model > 0) + 1e-10)
      loss /= 2

      



      
      loss.backward()
      torch.nn.utils.clip_grad_norm_(models.parameters(), args.max_grad_norm)
      optimizer.step()
      scheduler.step()  # Update learning rate schedule
      models.zero_grad()
      end_time = time.time() #datetime.datetime.now()
      program_running_time_seconds += end_time - start_time #(end_time - start_time).seconds
      
      global_step += 1

      if args.logging_steps > 0 and global_step % args.logging_steps == 0:
        ## model evaluation
        num_annotated_all.append(str(num_annotated))
        ## select a model for evaluation
        ranked_model_ids = ucb_action(args.num_of_experts)
        selected_model_id = ranked_model_ids[-1]
        result = evaluate(args, models[selected_model_id], tokenizer, language=args.eval_lang)
        f1_test.append(str(result['f1']))
        em_test.append(str(result['exact']))
        result["language"] = args.eval_lang
        logger.info("***** [Model %d ] Current Eval results *****", selected_model_id)
        for key in sorted(result.keys()):
          logger.info("  %s = %s", key, str(result[key]))



          
        logger.info("pro. of annotated: %lf", num_annotated / float(all_data_num) )
        logger.info("all_data_num: %d | num_annotated: %d | num_of_model_annotated: %d | num_of_model_same: %d", all_data_num, num_annotated, num_of_model_annotated, num_of_model_same)
        logger.info("estimated_rewards: %s", ' '.join([str(q) for q in list(estimated_rewards)]))
        logger.info("chosen_count: %s", ' '.join([str(q) for q in list(chosen_count)]))
        logger.info("co-estimated_rewards: %s", ' '.join([str(q) for q in list(co_estimated_rewards)]))
        logger.info("co-chosen_count: %s", ' '.join([str(q) for q in list(co_chosen_count)]))


        
        
  ## model evaluation
  num_annotated_all.append(str(num_annotated))
  ## select a model for evaluation
  ranked_model_ids = ucb_action(args.num_of_experts)
  selected_model_id = ranked_model_ids[-1]
  result = evaluate(args, models[selected_model_id], tokenizer, language=args.eval_lang)
  f1_test.append(str(result['f1']))
  em_test.append(str(result['exact']))
  result["language"] = args.eval_lang
  logger.info("***** [Model %d ] Current Eval results *****", selected_model_id)
  for key in sorted(result.keys()):
    logger.info("  %s = %s", key, str(result[key]))
        



  logger.info("pro. of annotated: %lf", num_annotated / float(all_data_num) )
  logger.info("all_data_num: %d | num_annotated: %d | num_of_model_annotated: %d | num_of_model_same: %d", all_data_num, num_annotated, num_of_model_annotated, num_of_model_same)
  logger.info("estimated_rewards: %s", ' '.join([str(q) for q in list(estimated_rewards)]))
  logger.info("chosen_count: %s", ' '.join([str(q) for q in list(chosen_count)]))
  logger.info("co-estimated_rewards: %s", ' '.join([str(q) for q in list(co_estimated_rewards)]))
  logger.info("co-chosen_count: %s", ' '.join([str(q) for q in list(co_chosen_count)]))



  print ("estimated_rewards: %s" % (' '.join([str(q) for q in list(estimated_rewards)])) )
  print ("chosen_count: %s" % ( ' '.join([str(q) for q in list(chosen_count)]) ))
  print ("combinations are as follows:")
  print (combins)
  print ("co-estimated_rewards: %s" % (' '.join([str(q) for q in list(co_estimated_rewards)])) )
  print ("co-chosen_count: %s" % ( ' '.join([str(q) for q in list(co_chosen_count)]) ))
  print ("result before adaptation f1:")
  print ('\n'.join(result_before_adaptation_f1))
  print ("result before adaptation em:")
  print ('\n'.join(result_before_adaptation_em))
  print ('\n\n')


  print ("pro. of annotated: %lf" % ( num_annotated / float(all_data_num)))
  print ('all_data_num: %d' % (all_data_num)  )
  print ('num_annotated_all:')
  print ('\n'.join(num_annotated_all))
  print ('\n\n')
  print ('f1_test:')
  print ('\n'.join(f1_test))
  print ('\n\n')
  print ('em_test:')
  print ('\n'.join(em_test))
  print ('\n\n')
  

  print ('---> time used is:')
  print ('%lf'  % (program_running_time_seconds) )

  return None #global_step, tr_loss / global_step

def evaluate(args, model, tokenizer, prefix="", language='en', lang2id=None):
  dataset, examples, features = load_and_cache_examples(args, tokenizer, evaluate=True, output_examples=True,
                              language=language, lang2id=lang2id)

  if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
    os.makedirs(args.output_dir)

  args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)


  #logger.info('testing at most 10000 samples')
  #dataset = dataset[:10000]

  # Note that DistributedSampler samples randomly
  eval_sampler = SequentialSampler(dataset)
  eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

  # multi-gpu evaluate
  if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
    model = torch.nn.DataParallel(model)

  # Eval!
  logger.info("***** Running evaluation {} *****".format(prefix))
  logger.info("  Num examples = %d", len(dataset))
  logger.info("  Batch size = %d", args.eval_batch_size)

  all_results = []
  start_time = timeit.default_timer()

  #for batch in tqdm(eval_dataloader, desc="Evaluating"):
  for batch in eval_dataloader:
    model.eval()
    batch = tuple(t.to(args.device) for t in batch)

    with torch.no_grad():
      inputs = {
        "input_ids": batch[0],
        "attention_mask": batch[1],
        "token_type_ids": None if args.model_type in ["xlm", "distilbert", "xlm-roberta"] else batch[2],
      }
      example_indices = batch[3]

      # XLNet and XLM use more arguments for their predictions
      if args.model_type in ["xlnet", "xlm"]:
        inputs.update({"cls_index": batch[4], "p_mask": batch[5]})
      if args.model_type == "xlm":
        inputs["langs"] = batch[6]

      outputs = model(**inputs)[:2]

    for i, example_index in enumerate(example_indices):
      eval_feature = features[example_index.item()]
      unique_id = int(eval_feature.unique_id)

      output = [to_list(output[i]) for output in outputs]

      # Some models (XLNet, XLM) use 5 arguments for their predictions, while the other "simpler"
      # models only use two.
      if len(output) >= 5:
        start_logits = output[0]
        start_top_index = output[1]
        end_logits = output[2]
        end_top_index = output[3]
        cls_logits = output[4]

        result = SquadResult(
          unique_id,
          start_logits,
          end_logits,
          start_top_index=start_top_index,
          end_top_index=end_top_index,
          cls_logits=cls_logits,
        )

      else:
        start_logits, end_logits = output
        result = SquadResult(unique_id, start_logits, end_logits)

      all_results.append(result)

  evalTime = timeit.default_timer() - start_time
  logger.info("  Evaluation done in total %f secs (%f sec per example)", evalTime, evalTime / len(dataset))

  # Compute predictions
  output_prediction_file = os.path.join(args.output_dir, "predictions_{}_{}.json".format(language, prefix))
  output_nbest_file = os.path.join(args.output_dir, "nbest_predictions_{}_{}.json".format(language, prefix))

  if args.version_2_with_negative:
    output_null_log_odds_file = os.path.join(args.output_dir, "null_odds_{}.json".format(prefix))
  else:
    output_null_log_odds_file = None

  # XLNet and XLM use a more complex post-processing procedure
  if args.model_type in ["xlnet", "xlm"]:
    start_n_top = model.config.start_n_top if hasattr(model, "config") else model.module.config.start_n_top
    end_n_top = model.config.end_n_top if hasattr(model, "config") else model.module.config.end_n_top

    predictions = compute_predictions_log_probs(
      examples,
      features,
      all_results,
      args.n_best_size,
      args.max_answer_length,
      output_prediction_file,
      output_nbest_file,
      output_null_log_odds_file,
      start_n_top,
      end_n_top,
      args.version_2_with_negative,
      tokenizer,
      args.verbose_logging,
    )
  else:
    """
    predictions = compute_predictions_logits(
      examples,
      features,
      all_results,
      args.n_best_size,
      args.max_answer_length,
      args.do_lower_case,
      output_prediction_file,
      output_nbest_file,
      output_null_log_odds_file,
      args.verbose_logging,
      args.version_2_with_negative,
      args.null_score_diff_threshold,
      tokenizer
    )
    """


    predictions = compute_predictions_logits(
      examples,
      features,
      all_results,
      args.n_best_size,
      args.max_answer_length,
      args.do_lower_case,
      output_prediction_file,
      output_nbest_file,
      output_null_log_odds_file,
      args.verbose_logging,
      args.version_2_with_negative,
      args.null_score_diff_threshold,
      tokenizer,
      #map_to_origin=not (args.model_type == "xlm-roberta" and (language == 'zh' or language == "ko")), ##888
    )


    """
    predictions = compute_predictions_logits(
      examples,
      features,
      all_results,
      args.n_best_size,
      args.max_answer_length,
      True,
      output_prediction_file,
      output_nbest_file,
      output_null_log_odds_file,
      args.verbose_logging,
      args.version_2_with_negative,
      args.null_score_diff_threshold,
      tokenizer,
      #map_to_origin=not (args.model_type == "xlm-roberta" and (language == 'zh' or language == "ko")), ##888
    )
    """


    

  # Compute the F1 and exact scores.
  results = squad_evaluate(examples, predictions)
  return results

def qa_loss(start_logits, end_logits, start_positions, end_positions, return_start_end=False):
  ignored_index = start_logits.size(1)
  start_positions.clamp_(0, ignored_index)
  end_positions.clamp_(0, ignored_index)

  loss_fct = CrossEntropyLoss(ignore_index=ignored_index, reduce=False)
  start_loss = loss_fct(start_logits, start_positions)
  end_loss = loss_fct(end_logits, end_positions)
  total_loss = (start_loss + end_loss) / 2

  if not return_start_end:
    return total_loss
  else:
    return total_loss, start_loss, end_loss

def load_and_cache_examples(args, tokenizer, evaluate=False, output_examples=False,
              language='en', lang2id=None):
  if args.local_rank not in [-1, 0] and not evaluate:
    # Make sure only the first process in distributed training process the dataset, and the others will use the cache
    torch.distributed.barrier()

  # Load data features from cache or dataset file
  input_dir = args.data_dir if args.data_dir else "."
  cached_features_file = os.path.join(
    input_dir,
    "cached_{}_{}_{}_{}".format(
      os.path.basename(args.predict_file) if evaluate else os.path.basename(args.train_file),
      'train' if not evaluate else 'eval',
      list(filter(None, args.model_name_or_path.split("/"))).pop(),
      str(args.max_seq_length),
      str(language)
    ),
  )

  # Init features and dataset from cache if it exists
  if os.path.exists(cached_features_file) and not args.overwrite_cache:
    logger.info("Loading features from cached file %s", cached_features_file)
    features_and_dataset = torch.load(cached_features_file)
    features, dataset = features_and_dataset["features"], features_and_dataset["dataset"]
  else:
    logger.info("Creating features from dataset file at %s", input_dir)

    if not args.data_dir and ((evaluate and not args.predict_file) or (not evaluate and not args.train_file)):
      try:
        import tensorflow_datasets as tfds
      except ImportError:
        raise ImportError("If not data_dir is specified, tensorflow_datasets needs to be installed.")

      if args.version_2_with_negative:
        logger.warn("tensorflow_datasets does not handle version 2 of SQuAD.")

      tfds_examples = tfds.load("squad")
      examples = SquadV1Processor().get_examples_from_dataset(tfds_examples, evaluate=evaluate, language=language)
    else:
      processor = SquadV2Processor() if args.version_2_with_negative else SquadV1Processor()
      if evaluate:
        examples = processor.get_dev_examples(args.data_dir, filename=args.predict_file, language=language)
      else:
        examples = processor.get_train_examples(args.data_dir, filename=args.train_file, language=language)

    features, dataset = squad_convert_examples_to_features(
      examples=examples,
      tokenizer=tokenizer,
      max_seq_length=args.max_seq_length,
      doc_stride=args.doc_stride,
      max_query_length=args.max_query_length,
      is_training=not evaluate,
      return_dataset="pt",
      threads=args.threads,
      lang2id=lang2id
    )

    if args.local_rank in [-1, 0]:
      logger.info("Saving features into cached file %s", cached_features_file)
      torch.save({"features": features, "dataset": dataset}, cached_features_file)

      print (cached_features_file)
      print (cached_features_file)
      print (cached_features_file)

  if output_examples:
    processor = SquadV2Processor() if args.version_2_with_negative else SquadV1Processor()
    if evaluate:
        examples = processor.get_dev_examples(args.data_dir, filename=args.predict_file, language=language)
    else:
        examples = processor.get_train_examples(args.data_dir, filename=args.train_file, language=language)



  if args.local_rank == 0 and not evaluate:
    # Make sure only the first process in distributed training process the dataset, and the others will use the cache
    torch.distributed.barrier()

  if output_examples:
    return dataset, examples, features
  return dataset


def main():
  parser = argparse.ArgumentParser()

  # Required parameters
  parser.add_argument(
    "--model_type",
    default=None,
    type=str,
    required=True,
    help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
  )
  parser.add_argument(
    "--model_name_or_path",
    default=None,
    type=str,
    required=True,
    help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS),
  )
  parser.add_argument(
    "--output_dir",
    default=None,
    type=str,
    required=True,
    help="The output directory where the model checkpoints and predictions will be written.",
  )

  # Other parameters
  parser.add_argument(
    "--data_dir",
    default=None,
    type=str,
    help="The input data dir. Should contain the .json files for the task."
    + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
  )
  parser.add_argument(
    "--train_file",
    default=None,
    type=str,
    help="The input training file. If a data dir is specified, will look for the file there"
    + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
  )
  parser.add_argument(
    "--predict_file",
    default=None,
    type=str,
    help="The input evaluation file. If a data dir is specified, will look for the file there"
    + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
  )
  parser.add_argument(
    "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name"
  )
  parser.add_argument(
    "--tokenizer_name",
    default="",
    type=str,
    help="Pretrained tokenizer name or path if not the same as model_name",
  )
  parser.add_argument(
    "--cache_dir",
    default="",
    type=str,
    help="Where do you want to store the pre-trained models downloaded from s3",
  )

  parser.add_argument(
    "--version_2_with_negative",
    action="store_true",
    help="If true, the SQuAD examples contain some that do not have an answer.",
  )
  parser.add_argument(
    "--null_score_diff_threshold",
    type=float,
    default=0.0,
    help="If null_score - best_non_null is greater than the threshold predict null.",
  )


  parser.add_argument(
    "--topk",
    type = float,
    default = 0.5
  )

  
  parser.add_argument(
    "--neg_reward",
    type = float,
    default = 0.
  )



  parser.add_argument(
    "--max_seq_length",
    default=384,
    type=int,
    help="The maximum total input sequence length after WordPiece tokenization. Sequences "
    "longer than this will be truncated, and sequences shorter than this will be padded.",
  )
  parser.add_argument(
    "--doc_stride",
    default=128,
    type=int,
    help="When splitting up a long document into chunks, how much stride to take between chunks.",
  )
  parser.add_argument(
    "--max_query_length",
    default=64,
    type=int,
    help="The maximum number of tokens for the question. Questions longer than this will "
    "be truncated to this length.",
  )
  parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
  parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
  parser.add_argument(
    "--evaluate_during_training", action="store_true", help="Run evaluation during training at each logging step."
  )
  parser.add_argument(
    "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
  )

  parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
  parser.add_argument(
    "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation."
  )
  parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
  parser.add_argument(
    "--gradient_accumulation_steps",
    type=int,
    default=1,
    help="Number of updates steps to accumulate before performing a backward/update pass.",
  )
  parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
  parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
  parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
  parser.add_argument(
    "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform."
  )
  parser.add_argument(
    "--max_steps",
    default=-1,
    type=int,
    help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
  )
  parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
  parser.add_argument(
    "--n_best_size",
    default=20,
    type=int,
    help="The total number of n-best predictions to generate in the nbest_predictions.json output file.",
  )
  parser.add_argument(
    "--max_answer_length",
    default=30,
    type=int,
    help="The maximum length of an answer that can be generated. This is needed because the start "
    "and end predictions are not conditioned on one another.",
  )
  parser.add_argument(
    "--verbose_logging",
    action="store_true",
    help="If true, all of the warnings related to data processing will be printed. "
    "A number of warnings are expected for a normal SQuAD evaluation.",
  )

  parser.add_argument("--logging_steps", type=int, default=50, help="Log every X updates steps.")
  parser.add_argument("--save_steps", type=int, default=50, help="Save checkpoint every X updates steps.")
  parser.add_argument(
    "--eval_all_checkpoints",
    action="store_true",
    help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
  )
  parser.add_argument("--no_cuda", action="store_true", help="Whether not to use CUDA when available")
  parser.add_argument(
    "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
  )
  parser.add_argument(
    "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
  )
  parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

  parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
  parser.add_argument(
    "--fp16",
    action="store_true",
    help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
  )
  parser.add_argument(
    "--fp16_opt_level",
    type=str,
    default="O1",
    help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
    "See details at https://nvidia.github.io/apex/amp.html",
  )
  parser.add_argument("--server_ip", type=str, default="", help="Can be used for distant debugging.")
  parser.add_argument("--server_port", type=str, default="", help="Can be used for distant debugging.")

  parser.add_argument("--threads", type=int, default=1, help="multiple threads for converting example to features")

  parser.add_argument("--train_lang", type=str, default="en", help="The language of the training data")
  parser.add_argument("--eval_lang", type=str, default="en", help="The language of the test data")


  ## HIL
  parser.add_argument("--model_name_or_path_large", type=str, help="")
  parser.add_argument("--layer_to_drop", default = 6, type = int)
  parser.add_argument("--drop_p", default = 0.5, type = float)
  parser.add_argument("--hil_weight", type=float, default=1., help='the weight of human feedback')
  parser.add_argument("--do_HIL_multi_arm_ucb_self_train", action="store_true")
  parser.add_argument("--do_HIL_multi_arm_ucb_co_train_preference_f1", action="store_true")

  
  
  parser.add_argument("--num_layer_to_update", type=int, default=2)
  parser.add_argument("--layer_to_drop_large", type=int, default=12, help='')
  parser.add_argument("--drop_p_large", type=float, default=0.5, help='')
  


  parser.add_argument("--training_data_num", type=int, default=0, help='training data num')
  parser.add_argument("--num_of_experts", type=int, default=3, help='the number of experts')
  parser.add_argument("--threshold", type=float, default=0.2, help='the threshold for deciding the certain data')

  parser.add_argument("--source_model_path", nargs='+', type=str)
  parser.add_argument("--leave_feedback_rate", type=float, help='the probability to leave feedback')
  parser.add_argument("--training_mode", choices = ['semi', 'super'] )
  parser.add_argument("--sample_num", type = int, default = 1 )
  parser.add_argument("--co_ucb", type = int, default = 1, help='1: use better version of co_ucb; 0: a baseline ' )
  parser.add_argument("--eval_before_adapt", type = int, default = 0, help='1: evaluation before adaptation;   0: no evaluation before adaptation ' )

  
  

  args = parser.parse_args()

  if (
    os.path.exists(args.output_dir)
    and os.listdir(args.output_dir)
    and args.do_train
    and not args.overwrite_output_dir
  ):
    raise ValueError(
      "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
        args.output_dir
      )
    )

  # Setup distant debugging if needed
  if args.server_ip and args.server_port:
    # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
    import ptvsd

    print("Waiting for debugger attach")
    ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
    ptvsd.wait_for_attach()

  # Setup CUDA, GPU & distributed training
  if args.local_rank == -1 or args.no_cuda:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
  else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    torch.distributed.init_process_group(backend="nccl")
    args.n_gpu = 1
  args.device = device

  # Setup logging
  logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
  )
  logger.warning(
    "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
    args.local_rank,
    device,
    args.n_gpu,
    bool(args.local_rank != -1),
    args.fp16,
  )

  # Set seed
  set_seed(args)

  # Load pretrained model and tokenizer
  if args.local_rank not in [-1, 0]:
    # Make sure only the first process in distributed training will download model & vocab
    torch.distributed.barrier()

  args.model_type = args.model_type.lower()
  config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
  config = config_class.from_pretrained(
    args.config_name if args.config_name else args.model_name_or_path,
    cache_dir=args.cache_dir if args.cache_dir else None,
  )
  config.layer_to_drop = args.layer_to_drop
  config.drop_p = args.drop_p
  # Set using of language embedding to True
  if args.model_type == "xlm":
    config.use_lang_emb = True
  tokenizer = tokenizer_class.from_pretrained(
    args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
    do_lower_case=args.do_lower_case,
    cache_dir=args.cache_dir if args.cache_dir else None,
  )
  model = model_class.from_pretrained(
    args.model_name_or_path,
    from_tf=bool(".ckpt" in args.model_name_or_path),
    config=config,
    cache_dir=args.cache_dir if args.cache_dir else None,
  )

  lang2id = config.lang2id if args.model_type == "xlm" else None
  logger.info("lang2id = {}".format(lang2id))

  if args.local_rank == 0:
    # Make sure only the first process in distributed training will download model & vocab
    torch.distributed.barrier()

  model.to(args.device)

  logger.info("Training/evaluation parameters %s", args)

  # Before we do anything with models, we want to ensure that we get fp16 execution of torch.einsum if args.fp16 is set.
  # Otherwise it'll default to "promote" mode, and we'll get fp32 operations. Note that running `--fp16_opt_level="O2"` will
  # remove the need for this code, but it is still valid.
  if args.fp16:
    try:
      import apex

      apex.amp.register_half_function(torch, "einsum")
    except ImportError:
      raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

  # Training
  if args.do_train:
    train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False, output_examples=False, language=args.train_lang, lang2id=lang2id)
    global_step, tr_loss = train(args, train_dataset, model, tokenizer)
    logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

  # Save the trained model and the tokenizer
  if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
    # Create output directory if needed
    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
      os.makedirs(args.output_dir)

    logger.info("Saving model checkpoint to %s", args.output_dir)
    # Save a trained model, configuration and tokenizer using `save_pretrained()`.
    # They can then be reloaded using `from_pretrained()`
    # Take care of distributed/parallel training
    model_to_save = model.module if hasattr(model, "module") else model
    model_to_save.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Good practice: save your training arguments together with the trained model
    torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

    # Load a trained model and vocabulary that you have fine-tuned
    model = model_class.from_pretrained(args.output_dir, force_download=True)
    tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
    model.to(args.device)

  # Evaluation - we can ask to evaluate all the checkpoints (sub-directories) in a directory
  results = {}
  if args.do_eval and args.local_rank in [-1, 0]:
    if args.do_train:
      logger.info("Loading checkpoints saved during training for evaluation")
      checkpoints = [args.output_dir]
      if args.eval_all_checkpoints:
        checkpoints = list(
          os.path.dirname(c)
          for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
        )
        logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce model loading logs
    else:
      logger.info("Loading checkpoint %s for evaluation", args.model_name_or_path)
      checkpoints = [args.model_name_or_path]

    logger.info("Evaluate the following checkpoints: %s", checkpoints)

    for checkpoint in checkpoints:
      # Reload the model
      global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
      model = model_class.from_pretrained(checkpoint, force_download=True)
      model.to(args.device)

      # Evaluate
      result = evaluate(args, model, tokenizer, prefix=global_step, language=args.eval_lang, lang2id=lang2id)

      result = dict((k + ("_{}".format(global_step) if global_step else ""), v) for k, v in result.items())
      results.update(result)

  logger.info("Results: {}".format(results))

  results = {}
  if (  args.do_HIL_multi_arm_ucb_self_train  or  args.do_HIL_multi_arm_ucb_co_train_preference_f1 ) and args.local_rank in [-1, 0]:
    if args.do_train:
      logger.info("Loading checkpoints saved during training for evaluation")
      checkpoints = [args.output_dir]
      if args.eval_all_checkpoints:
        checkpoints = list(
          os.path.dirname(c)
          for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
        )
        logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce model loading logs
    else:
      logger.info("Loading checkpoint %s for evaluation", args.model_name_or_path)
      checkpoints = [args.model_name_or_path]

    logger.info("Evaluate the following checkpoints: %s", checkpoints)

    for checkpoint in checkpoints:
      # Reload the model
      global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
      
      model = model_class.from_pretrained(checkpoint, config = config, force_download=True)
      model.to(args.device)

      # Evaluate
      if args.do_HIL_multi_arm_ucb_self_train:
        HIL_multi_arm_ucb_self_train(args, tokenizer, config, lang2id)
        print ('\n\n')


      elif args.do_HIL_multi_arm_ucb_co_train_preference_f1:
        HIL_multi_arm_ucb_co_train_preference_f1(args, tokenizer, config, lang2id)
        print ('\n\n')


    

      
        
  

  return results


if __name__ == "__main__":
  main()



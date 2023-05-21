#!/bin/bash
# Copyright 2020 Google and DeepMind.
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

# Script to train a model on SQuAD v1.1 or the English TyDiQA-GoldP train data.

REPO=$PWD
MODEL=${1:-xlm-roberta-base}
#MODEL=${1:-bert-base-multilingual-cased}
SRC=${2:-squad}
TGT=${3:-mrqa}
GPU=${4:-1}
DATA_DIR=${5:-"$REPO/download/"}
OUT_DIR=${6:-"$REPO/outputs/"}

for seed in  1 2  3  ; do
for topk in  1    ; do

training_data_num=100000
neg_reward=0 #
leave_feedback_rate=1          
training_mode='super'
threshold=5.5
num_of_experts=1
co_ucb=1                      
eval_before_adapt=0         

langs=( 1 2 4 5 3 6 )


num_layer_to_update=12 
layer_to_drop=12
drop_p=0


hil_weight=1           
logging_steps=1000000000 
LR=5e-7
BATCH_SIZE=16 


MAXL=384
NUM_EPOCHS=1 #10 #20 #3.0
if [ $MODEL == "bert-base-multilingual-cased" ]; then
  MODEL_TYPE="bert"
elif [ $MODEL == "xlm-mlm-100-1280" ] || [ $MODEL == "xlm-mlm-tlm-xnli15-1024" ]; then
  MODEL_TYPE="xlm"
elif [ $MODEL == "xlm-roberta-large" ] || [ $MODEL == "xlm-roberta-base" ]; then
  MODEL_TYPE="xlm-roberta"
fi




#langs=( SQuAD HotpotQA NaturalQA NewsQA TriviaQA )

echo "************************"
echo ${MODEL}
echo "************************"

echo
for lang in ${langs[@]}; do
  echo "  $lang "
  
  DIR=${DATA_DIR}/${lang,,}
  if [ $lang == '1' ]; then
    ## SQuAD, NewsQA, NaturalQA, TriviaQA, SearchQA --> HotpotQA
    source_0=./outputs/squad/xlm-roberta-base_LR3e-5_EPOCH3.0_maxlen384_batchsize4_gradacc8
    TRAIN_FILE=${DATA_DIR}/hotpotqa/HotpotQA-train-from-MRQA.json    
    PREDICT_FILE=${DATA_DIR}/hotpotqa/HotpotQA-dev-from-MRQA.json
    data_dir=${DATA_DIR}/hotpotqa/



  
  elif [ $lang == '2' ]; then
    ## HotpotQA, NewsQA, NaturalQA, TriviaQA, SearchQA --> SQuAD
    source_0=./outputs/newsqa/xlm-roberta-base_LR3e-5_EPOCH3.0_maxlen384_batchsize32_gradacc1
    TRAIN_FILE=${DATA_DIR}/squad/train-v1.1.json
    PREDICT_FILE=${DATA_DIR}/squad/dev-v1.1.json
    data_dir=${DATA_DIR}/squad/




  elif [ $lang == '3' ]; then   ## not good
    ## SQuAD, HotpotQA, NewsQA, NaturalQA, SearchQA -->  TriviaQA
    source_0=./outputs/searchqa/xlm-roberta-base_LR3e-5_EPOCH3.0_maxlen384_batchsize32_gradacc1
    TRAIN_FILE=${DATA_DIR}/triviaqa/TriviaQA-web-train-from-MRQA.json 
    PREDICT_FILE=${DATA_DIR}/triviaqa/TriviaQA-web-dev-from-MRQA.json 
    data_dir=${DATA_DIR}/triviaqa/




  elif [ $lang == '4' ]; then
    ## SQuAD, HotpotQA, NewsQA, TriviaQA, SearchQA --> NaturalQA
    source_0=./outputs/newsqa/xlm-roberta-base_LR3e-5_EPOCH3.0_maxlen384_batchsize32_gradacc1
    TRAIN_FILE=${DATA_DIR}/naturalqa/NaturalQuestionsShort-train-from-MRQA.json
    PREDICT_FILE=${DATA_DIR}/naturalqa/NaturalQuestionsShort-dev-from-MRQA.json
    data_dir=${DATA_DIR}/naturalqa/

  elif [ $lang == '5' ]; then
    ## SQuAD, HotpotQA, NaturalQA, TriviaQA, SearchQA --> NewsQA
    source_0=./outputs/squad/xlm-roberta-base_LR3e-5_EPOCH3.0_maxlen384_batchsize4_gradacc8
    TRAIN_FILE=${DATA_DIR}/newsqa/NewsQA-train-from-MRQA.json 
    PREDICT_FILE=${DATA_DIR}/newsqa/NewsQA-dev-from-MRQA.json 
    data_dir=${DATA_DIR}/newsqa/





  elif [ $lang == '6' ]; then
    ## SQuAD, HotpotQA, NaturalQA, TriviaQA, NewsQA --> SearchQA
    source_0=./outputs/triviaqa/xlm-roberta-base_LR3e-5_EPOCH3.0_maxlen384_batchsize32_gradacc1
    TRAIN_FILE=${DATA_DIR}/searchqa/SearchQA-train-from-MRQA.json
    PREDICT_FILE=${DATA_DIR}/searchqa/SearchQA-dev-from-MRQA.json
    data_dir=${DATA_DIR}/searchqa/
  fi






  PREDICTIONS_DIR=${lang}/
  PRED_DIR=${PREDICTIONS_DIR}/
  mkdir -p "${PRED_DIR}"


  CUDA_VISIBLE_DEVICES=${GPU} python third_party/run_squad.py \
    --model_type ${MODEL_TYPE} \
    --model_name_or_path ${source_0} \
    --do_HIL_multi_arm_ucb_self_train \
    --num_of_experts $num_of_experts \
    --threshold $threshold \
    --topk $topk \
    --neg_reward $neg_reward \
    --seed $seed \
    --learning_rate $LR \
    --per_gpu_eval_batch_size $BATCH_SIZE \
    --per_gpu_train_batch_size $BATCH_SIZE \
    --num_train_epochs $NUM_EPOCHS \
    --logging_steps $logging_steps \
    --eval_lang en \
    --training_data_num $training_data_num \
    --layer_to_drop $layer_to_drop \
    --drop_p $drop_p \
    --hil_weight $hil_weight \
    --num_layer_to_update $num_layer_to_update \
    --output_dir "${PRED_DIR}" \
    --leave_feedback_rate $leave_feedback_rate \
    --training_mode $training_mode \
    --source_model_path $source_0 \
    --train_file ${TRAIN_FILE} \
    --predict_file ${PREDICT_FILE} \
    --co_ucb  $co_ucb \
    --eval_before_adapt  $eval_before_adapt \
    --data_dir $data_dir


  






echo "source: $lang"
echo "tgt: ${PREDICT_FILE}"
echo "leave_feedback_rate: $leave_feedback_rate"
echo "training_mode: $training_mode"
echo "hil: $hil"
echo "num_of_experts: $num_of_experts"
echo "co_ucb: $co_ucb"
echo "eval_before_adapt: $eval_before_adapt"
echo "threshold: $threshold"
echo "training_data_num: $training_data_num"
echo "neg_reward: $neg_reward"
echo "topk: $topk"
echo "layer_to_drop: $layer_to_drop"
echo "drop_p: $drop_p"
echo "hil_weight: $hil_weight"
echo "num_layer_to_update: $num_layer_to_update"
echo "layer_to_drop_large: $layer_to_drop_large"
echo "drop_p_large: $drop_p_large"
echo "logging_steps: $logging_steps"
echo "BATCH_SIZE: $BATCH_SIZE"
echo "LR: $LR"
echo "seed: $seed"



done


done
done



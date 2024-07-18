#! /bin/bash

CACHE_PATH=/home/ycshi/sticker-dialogue-summ/results/bart  #your cache path
TRAIN_DATA=/home/ycshi/sticker-dialogue-summ/datas/jsonl/train.jsonl
VAL_DATA=/home/ycshi/sticker-dialogue-summ/datas/jsonl/dev.jsonl
TEST_DATA=/home/ycshi/sticker-dialogue-summ/datas/jsonl/test.jsonl   #your data path in jsonl style
MODEL=/home/ycshi/huggingface/fnlpbart-large-chinese  #your model path :bart T5 pegasus  mbart....
RESYLT_NAME=bart #your result file name


#my_array=(100 200 300 400 500 600 700 800 900 1000)


#for element in "${my_array[@]}"
#do
    #echo $element
python /train_with_finetune.py \
    --train_data_path $TRAIN_DATA \
    --valid_data_path $VAL_DATA \
    --test_data_path $TEST_DATA \
    --cache_path $CACHE_PATH \
    --current_project_name descbacksent \
    --result_name $RESYLT_NAME \
    --train_data_nums 4421 \   #train_set_size
    --num_epoch 200 \
    --project_dir $CACHE_PATH \
    --pretrain_model $MODEL \
    --cache
#done

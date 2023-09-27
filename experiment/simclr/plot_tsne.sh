#!/bin/bash

DATASET=CIFAR10
BATCH_SIZE=5000
TRAIN_BATCH_SIZE=128
EPOCH=500
LOSS="dclw"
TEMP=0.07
python embeddings.py \
  --batch_size $BATCH_SIZE \
  --dataset $DATASET \
  --loss $LOSS \
  --model_path "../../results/checkpoints/${DATASET}_128_${TEMP}_200_${TRAIN_BATCH_SIZE}_${EPOCH}_${LOSS}_model.pth"
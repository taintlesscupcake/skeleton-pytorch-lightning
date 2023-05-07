#!/bin/bash

echo "Skeleton Training Script"

name="skeleton_train"
trainset="example_train"
testset="example_test"
train_data_path="data/${trainset}/"
test_data_path="data/${testset}/"
epochs=100
batch_size=32
lr=0.001
num_workers=4
description="Skeleton Training Script"
# Add more arguments as you need

# Specify the visible GPUs, so that the GPUs can be limited
CUDA_VISIBLE_DEVICES=0,1 python3 train.py --name ${name} \
                                         --trainset ${trainset} \
                                         --testset ${testset} \
                                         --train_data_path ${train_data_path} \
                                         --test_data_path ${test_data_path} \
                                         --epochs ${epochs} \
                                         --batch_size ${batch_size} \
                                         --lr ${lr} \
                                         --num_workers ${num_workers} \
                                         --description ${description}
                                            # Add more arguments as you need

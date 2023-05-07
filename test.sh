#!/bin/bash

echo "Skeleton Testing Script"

name="skeleton_train"
trainset="example_train"
testset="example_test"
train_data_path="data/${trainset}/"
test_data_path="data/${testset}/"
num_workers=4
description="Skeleton Testing Script"
# Add more arguments as you need

# Specify the visible GPUs, so that the GPUs can be limited
CUDA_VISIBLE_DEVICES=0,1 python3 test.py --name ${name} \
                                         --trainset ${trainset} \
                                         --testset ${testset} \
                                         --train_data_path ${train_data_path} \
                                         --test_data_path ${test_data_path} \
                                         --num_workers ${num_workers} \
                                         --description ${description} \
                                            # Add more arguments as you need
#!/bin/bash/python
echo 'run program start'
export ROOT_PATH=$(pwd)
echo "root_path=${ROOT_PATH}"
cd $ROOT_PATH

source ${ROOT_PATH}/textCNN/env.sh
${PYTHON} --version

export BASE_DIR=${ROOT_PATH}/textCNN/src/
Data_Path=${ROOT_PATH}/data/input
mkdir ${ROOT_PATH}/data/input-fenci
mkdir ${ROOT_PATH}/data/input-samples
mkdir ${ROOT_PATH}/trained_model/text_cnn

rm -rf ${ROOT_PATH}/data/input-samples/*

# 如果该路径下模型文件存在，则会读取模型进行增量训练
Save_Path=${ROOT_PATH}/trained_model/textCNN/
mkdir ${Save_Path}


${PYTHON} ${BASE_DIR}/train_textCNN.py \
   --traning_data_path=${Data_Path} \
   --ckpt_dir=${Save_Path} \
   --sentence_len=128 \
   --batch_size=64 \
   --learning_rate=0.003 \
   --num_epochs=50 \
   --threshold=0.5 \
   --is_fenci=True \
   --labels=label \
   --multi_label_flag=True

echo 'run program finish'

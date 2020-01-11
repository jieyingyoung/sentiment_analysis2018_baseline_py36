#!/user/bin/env python
# -*- coding:utf-8 -*-

import os
data_path = os.path.abspath('') + "/data"
model_path = data_path + "/model/"
train_data_path = data_path + "/train/sentiment_analysis_trainingset.csv" # 训练集文件存放路径
validate_data_path = data_path + "/valid/sentiment_analysis_validationset.csv" # 验证集文件存放路径
test_data_path = data_path + "/test/sentiment_analysis_testa.csv" #测试集文件存放路径
test_data_predict_output_path = data_path + "/predict/testa_predict.csv" #测试集预测结果文件存放路径
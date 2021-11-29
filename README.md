# face_attribute
shisuo face_attribute

### 批量测试
```shell
python infer_main.py [--batch_size ${BATCH SIZE NUMBER}] [--num_workers ${NUM_WORKERS NUMBER}] [--resume_path ${CHECKPOINT_FILE}] [--test_data_path ${TEST_DATA_PATH}] [-c] [-s] [--save_path ${SAVE_PATH}]
```

举个栗子：
```shell
python infer_main.py --batch_size 128 --num_workers 4 --resume_path checkpoint/test.pth --test_data_path test_data/rap_test -c -s --save_path work_dir/rap_test_output_img
```

#### 可选参数
- `--batch_size`: 批量大小，可根据显存大小调整（默认32，最好为2的倍数，如16、32、64、128、256等）；
- `--num_workers`: PyTorch的DataLoader中读取数据的线程数；
- `--resume_path`: 模型参数文件的路径；
- `--test_data_path`: 测试数据的存放路径；
- `-c/--confidence`: 输出预测属性的置信度；
- `-s/--show`: 是否将属性预测结果打印到图片中；
- `--save_path`: 指定输出具有属性预测结果的图片路径。

### 指标测试
```shell
python F1_main.py [--batch_size ${BATCH SIZE NUMBER}] [--num_workers ${NUM_WORKERS NUMBER}] [--resume_path ${CHECKPOINT_FILE}] [--test_data_path ${TEST_DATA_PATH}] [--test_data_label ${TEST_DATA_LABEL}]
```

举个栗子：
```shell
python F1_main.py --batch_size 128 --num_workers 4 --resume_path checkpoint/test.pth --test_data_path test_data/rap_test --test_data_label data_list/my_rap2/test.txt
```


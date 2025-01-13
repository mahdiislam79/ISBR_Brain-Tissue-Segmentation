# ISBR_Brain-Tissue-Segmentation

## 1. Requirements
- Ensure Python 3.8 or higher is installed.
- Install the required packages:
  ```bash
  pip install torch torchvision numpy nibabel scipy pandas wandb

## 2.Dataset
- Ensure the dataset is in NIfTI format `(*.nii.gz)`.
- Organize your dataset into the following structure:
```bash
./data/
    Training_Set/
    Validation_Set/
    Test_set/
```

## 3. Training
- Run the training script to train the model:
```bash
python main.py --data_path <path_to_data> \
               --network_name <model_name> \
               --learning_rate <lr> \
               --epochs <num_epochs> \
               --batch_size <batch_size>
```

## 4. Inference
- Run the inference script to validate or test the model:
```bash
python inference.py --model_path <path_to_model> \
                    --data_dir <path_to_data> \
                    --set_type <val/test> \
                    --device <cpu/cuda>
```

# finetune-whisper

- https://huggingface.co/blog/fine-tune-whisper
- https://huggingface.co/openai/whisper-large-v3

# Persian ASR with Whisper

This project trains an automatic speech recognition (ASR) model on Persian audio using Whisper from Anthropic.

## Datasets

The following datasets are used for training and evaluation:

- Common Voice Persian
- CRM Persian
- KYC Persian

They are concatenated into a single training set and evaluation set.

## Model

- WhisperForConditionalGeneration from Whisper is used as the model
- Training starts from a pre-trained Whisper base model
- Training arguments:
  - Batch size: 16
  - Learning rate: 1e-5
  - Num epochs: 3

## Training

To run training:

```
./multigpu.sh
```

The training script handles logging with MLflow, metrics computation, and model saving.

Evaluation is performed on a subset of the test set during training.

## Evaluation

The Word Error Rate (WER) metric is used for evaluation.

Prediction and label texts are saved to `prediction.txt` and `labels.txt` respectively for analysis.

## Processing

To save time, the original training process is split in two sections: 
1- Data Preprocessing: `Preprocess_CommonVoice.ipynb`, `Preprocess_CRM_KYC.ipynb` -> Whisper Processed Data
2- Training: It only uses generated processed data to train the model 

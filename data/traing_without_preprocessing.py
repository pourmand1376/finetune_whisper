#!/usr/bin/env python
# coding: utf-8

# In[1]:

from datasets import concatenate_datasets
from datasets import load_dataset, DatasetDict
from accelerate import Accelerator
import mlflow
# In[2]:

mlflow.set_tracking_uri('http://10.1.20.6:5555')
mlflow.set_experiment('ASR Whisper')
mlflow.transformers.autolog()
mlflow.autolog()
mlflow.pytorch.autolog()

common_voice=load_dataset('dataset/whisper_processed_data/common_voice_processed.hf',
                         cache_dir='dataset/whisper_processed_data/.cache')
crm = load_dataset('dataset/whisper_processed_data/crm-processed.hf',
                  cache_dir='dataset/whisper_processed_data/.cache')
kyc= load_dataset('dataset/whisper_processed_data/kyc-processed.hf',
                 cache_dir='dataset/whisper_processed_data/.cache')

combined_data = DatasetDict()
combined_data['train']=concatenate_datasets([kyc['train'], common_voice['train'], crm['train']])
combined_data['test']=concatenate_datasets([kyc['test'], common_voice['test'], crm['test']])

# In[3]:


combined_data


# In[4]:


import torch

from dataclasses import dataclass
from typing import Any, Dict, List, Union

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


# In[5]:


from transformers import WhisperProcessor

processor = WhisperProcessor.from_pretrained("openai/whisper-medium", language="Persian", task="transcribe",cache_dir='v3')

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

from transformers import WhisperFeatureExtractor

feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-medium",cache_dir='v3')

from transformers import WhisperTokenizer

tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-medium", language="Persian", task="transcribe",cache_dir='v3')
# In[6]:


import evaluate

metric = evaluate.load("wer")


# In[7]:


def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    
    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


# In[8]:


from transformers import WhisperForConditionalGeneration

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-medium",cache_dir='v3')


# In[9]:


model.config.forced_decoder_ids = None
model.config.suppress_tokens = []


# In[10]:


from transformers import Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-large-medium-fa",  # change to a repo name of your choice
    per_device_train_batch_size=24,
    gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
    learning_rate=1e-5,
    warmup_ratio=0.1,
    #warmup_steps=500,
    num_train_epochs=5.0,
    #max_steps=4000,
    gradient_checkpointing=True,
    fp16=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=2000,
    eval_steps=2000,
    logging_steps=25,
    report_to="all",
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
)


# In[11]:


from transformers import Seq2SeqTrainer

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=combined_data["train"],
    eval_dataset=combined_data["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)


# In[ ]:
# accelerator = Accelerator(log_with="mlflow")

# if accelerator.is_main_process:
#     accelerator.init_trackers(
#         "ASR Whisper"
#     )

# my_trainer=accelerator.prepare(trainer)

# my_trainer.train()

with mlflow.start_run():
    trainer.train()
#trainer.train(resume_from_checkpoint=True)

import os
from datetime import datetime

# Get the current date and time
current_time = datetime.now()

# Format the date and time as a string (optional, you can customize the format)
formatted_time = current_time.strftime("%Y-%m-%d_%H-%M-%S")

trainer.save_model(f'whisper_best_{formatted_time}')


# In[ ]:





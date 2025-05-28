from datasets.dataset_dict import DatasetDict as ddict
# from pyarrow.dataset import dataset
from datasets.arrow_dataset import Dataset
import pyarrow as pa
from transformers import DataCollatorWithPadding
from transformers.hf_argparser import HfArg
import json
import os
from transformers import (
    BertForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    PreTrainedTokenizer,
    TrainingArguments,
    Trainer,
)
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
import evaluate
import numpy as np

def get_datasets(dir):
    res = []
    os.chdir(dir)
    files = os.listdir()
    for file in files:
        if not os.path.isdir(file):
            res.append(dir+'/'+file)
        else:
            sub_files = get_datasets(file)
            for sub_file in sub_files:
                res.append(dir+'/'+sub_file)
    os.chdir('..')
    return res
dataset_names = get_datasets('face2_zh_json')
def modify_dataset(names):
    sentence = []
    labels = []
    for name in names:
        raw = json.load(open(name))
        if 'human' in name:
            for i in range(len(raw)):
                sentence.append(raw[i]['input']+'[SEP]'+raw[i]['output'])
                labels.append(0)
        else:
            for i in range(len(raw['output'])):
                sentence.append(raw['input'][str(i)]+'[SEP]'+raw['output'][str(i)])
                labels.append(1)
    table= pa.table(
            pa.array( [{'text': data,
                         'label': label } for data,label in zip(sentence,labels)],
            type=pa.struct([('text',pa.string()),
                            ('label',pa.int64())])
            )
    )
    return Dataset(table)
tokenizer = AutoTokenizer.from_pretrained('model')
def choose_name(names,types):
    res = []
    for t in types.split('-'):
        for name in names:
            if t in name:
                res.append(name)
    return res
types = ['news','webnovel','wiki','webnovel-wiki','news-wiki','news-webnovel']
datas = ddict({t:modify_dataset(choose_name(dataset_names,t)) for t in types})
datas = datas.map(lambda examples:tokenizer(examples['text'], truncation=True),batched=True)
print('======================Data Loaded======================')
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
model = AutoModelForSequenceClassification.from_pretrained("model", num_labels=2)
print('======================Model Loaded======================')

choices = [(0,3),(1,4),(2,5),(3,0),(4,1),(5,2)]
for choice in choices:
    training_args = TrainingArguments(
        output_dir=f'Trained on {types[choice[0]]} Evaluated on {types[choice[1]]}',
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=10,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
    )
    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")
    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load("recall")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)  # 获取预测类别
        
        # 计算每个指标
        results = {}
        results.update(accuracy_metric.compute(predictions=predictions, references=labels))
        results.update(f1_metric.compute(predictions=predictions, references=labels, average="binary"))
        results.update(precision_metric.compute(predictions=predictions, references=labels, average="binary"))
        results.update(recall_metric.compute(predictions=predictions, references=labels, average="binary"))
        
        return results
    print('======================Initialized======================')

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datas[types[choice[0]]],
        eval_dataset=datas[types[choice[1]]],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    print('======================Training Begin======================')
    trainer.train()
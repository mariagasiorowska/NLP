print("SCRIPT STARTED")

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    set_seed
)

import numpy as np
from seqeval.metrics import precision_score, recall_score, f1_score

#config
MODEL_CHECKPOINT = "distilbert-base-uncased" 
OUTPUT_DIR = "./results"

TRAIN_BATCH_SIZE = 2
EVAL_BATCH_SIZE = 2
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3
WEIGHT_DECAY = 0.01

set_seed(42)

#loading data
print("Loading dataset...")
dataset = load_dataset("conll2003")

#small subset
dataset["train"] = dataset["train"].select(range(1000))
dataset["validation"] = dataset["validation"].select(range(200))
dataset["test"] = dataset["test"].select(range(200))

label_list = dataset["train"].features["ner_tags"].feature.names
num_labels = len(label_list)

print("Dataset loaded")

#tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

#tokenization and alignment
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        max_length=128,
        is_split_into_words=True
    )

    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []

        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)

            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

print("Tokenizing...")
tokenized_datasets = dataset.map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=dataset["train"].column_names
)
print("Tokenization done")

#model
print("Loading model...")
model = AutoModelForTokenClassification.from_pretrained(
    MODEL_CHECKPOINT,
    num_labels=num_labels,
    id2label={i: l for i, l in enumerate(label_list)},
    label2id={l: i for i, l in enumerate(label_list)}
)

data_collator = DataCollatorForTokenClassification(tokenizer)

#metrics
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=2)

    true_predictions = []
    true_labels = []

    for pred, lab in zip(predictions, labels):
        curr_preds = []
        curr_labels = []

        for p, l in zip(pred, lab):
            if l != -100:
                curr_preds.append(label_list[p])
                curr_labels.append(label_list[l])

        true_predictions.append(curr_preds)
        true_labels.append(curr_labels)

    return {
        "precision": precision_score(true_labels, true_predictions),
        "recall": recall_score(true_labels, true_predictions),
        "f1": f1_score(true_labels, true_predictions)
    }

#trainings
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=EVAL_BATCH_SIZE,
    num_train_epochs=NUM_EPOCHS,
    weight_decay=WEIGHT_DECAY,
    eval_strategy="epoch",   #transformers v5
    logging_strategy="steps",
    logging_steps=10,
    save_strategy="no",
    report_to="none",
    fp16=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

print("Starting training...")
trainer.train()

#final evaluation
print("Evaluating...")
predictions, labels, _ = trainer.predict(tokenized_datasets["test"])
results = compute_metrics((predictions, labels))

print("FINAL RESULTS")
for k, v in results.items():
    print(k, v)

#saving predictions as .iob2
print("Saving predictions to predictions.iob2...")

#running prediction on validation set
predictions, labels, _ = trainer.predict(tokenized_datasets["validation"])
predictions = np.argmax(predictions, axis=2)

#original (non-tokenized) validation data
original_dataset = dataset["validation"]

with open("predictions.iob2", "w") as f:
    for i, (pred, lab) in enumerate(zip(predictions, labels)):
        tokens = original_dataset[i]["tokens"]

        curr_preds = []
        for p, l in zip(pred, lab):
            if l != -100:
                curr_preds.append(label_list[p])

        #safety check
        assert len(tokens) == len(curr_preds), f"Mismatch at sentence {i}"

        for token, pred_label in zip(tokens, curr_preds):
            f.write(f"{token} {pred_label}\n")

        f.write("\n")

print("Predictions saved to predictions.iob2")
import os
import argparse
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch

from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    set_seed
)
from datasets import Dataset
import numpy as np
from seqeval.metrics import precision_score, recall_score, f1_score

# ── config ────────────────────────────────────────────────────────────────────
MODEL_CHECKPOINT = "distilbert-base-uncased"
OUTPUT_DIR       = "./results"

TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE  = 8
LEARNING_RATE    = 2e-5
NUM_EPOCHS       = 3
WEIGHT_DECAY     = 0.01

TRAIN_FILE       = "en_ewt-ud-train.iob2"
DEV_FILE         = "en_ewt-ud-dev.iob2"
TEST_FILE        = "en_ewt-ud-test-masked.iob2"
OUTPUT_FILE      = "predictions.iob2"
TEST_OUTPUT_FILE = "test_predictions.iob2"

parser = argparse.ArgumentParser(description="Transformer NER baseline on EWT")
parser.add_argument("--cpu", action="store_true", help="Force CPU training")
parser.add_argument(
    "--mps",
    action="store_true",
    help="Try Apple Metal GPU; on Mac the default is CPU for stability",
)
args = parser.parse_args()

# label set — EWT only uses these 7 labels
LABEL_LIST = ["O", "B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG"]
LABEL2ID   = {l: i for i, l in enumerate(LABEL_LIST)}
ID2LABEL   = {i: l for i, l in enumerate(LABEL_LIST)}


def read_ewt(path):
    """
    Reads an EWT .iob2 file and returns a list of dicts:
        {"tokens": [...], "ner_tags": [...]}

    Tab-separated rows: parts[0] = 1-based word index, parts[1] = token,
    parts[2] = IOB2 label. Lines starting with # are comments; empty lines
    are sentence boundaries.
    """
    sentences = []
    tokens, tags = [], []

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")

            # comment lines
            if line.startswith("#"):
                continue

            # empty line = sentence boundary
            if line.strip() == "":
                if tokens:
                    sentences.append({"tokens": tokens, "ner_tags": tags})
                    tokens, tags = [], []
                continue

            parts = line.split("\t")
            token = parts[1]
            label = parts[2]

            # map label to int; unknown labels fall back to O
            tag_id = LABEL2ID.get(label, LABEL2ID["O"])
            tokens.append(token)
            tags.append(tag_id)

    # last sentence if file doesn't end with blank line
    if tokens:
        sentences.append({"tokens": tokens, "ner_tags": tags})

    return sentences


def write_iob2_with_predictions(
    source_path, sentences_data, prediction_logits, output_path, tokenizer, label_list
):
    """
    Read source IOB2 (comments and token lines), replace the NER column (index 2)
    with first-subword predictions. sentences_data must align in order with
    sentence boundaries in source_path.
    """
    pred_ids = np.argmax(prediction_logits, axis=2)
    n_sent = len(sentences_data)

    with open(source_path, encoding="utf-8") as f_in, open(output_path, "w", encoding="utf-8") as f_out:
        sent_idx = 0
        for line in f_in:
            line = line.rstrip("\n")

            if line.startswith("#"):
                f_out.write(line + "\n")
                continue

            if line.strip() == "":
                f_out.write("\n")
                sent_idx += 1
                continue

            parts = line.split("\t")
            if sent_idx >= n_sent:
                raise ValueError(
                    f"More token lines in {source_path} than sentences in aligned data ({n_sent})"
                )

            sentence_preds = pred_ids[sent_idx]
            tokens_in_sent = sentences_data[sent_idx]["tokens"]
            encoding = tokenizer(
                tokens_in_sent,
                truncation=True,
                max_length=128,
                is_split_into_words=True,
            )
            word_ids = encoding.word_ids()
            seen = set()
            word_pred_map = {}
            for pos, wid in enumerate(word_ids):
                if wid is not None and wid not in seen:
                    seen.add(wid)
                    word_pred_map[wid] = label_list[sentence_preds[pos]]

            word_pos = int(parts[0]) - 1
            predicted_label = word_pred_map.get(word_pos, "O")
            parts[2] = predicted_label
            f_out.write("\t".join(parts) + "\n")


# ── metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=2)

    true_predictions, true_labels = [], []
    for pred, lab in zip(predictions, labels):
        curr_preds, curr_labels = [], []
        for p, l in zip(pred, lab):
            if l != -100:
                curr_preds.append(LABEL_LIST[p])
                curr_labels.append(LABEL_LIST[l])
        true_predictions.append(curr_preds)
        true_labels.append(curr_labels)

    return {
        "precision": precision_score(true_labels, true_predictions),
        "recall":    recall_score(true_labels, true_predictions),
        "f1":        f1_score(true_labels, true_predictions),
    }


def main():
    print("SCRIPT STARTED")
    set_seed(42)

    # ── data loading ──────────────────────────────────────────────────────────
    print("Loading EWT data...")
    train_data = read_ewt(TRAIN_FILE)
    dev_data   = read_ewt(DEV_FILE)
    test_data  = read_ewt(TEST_FILE)

    train_dataset = Dataset.from_list(train_data)
    dev_dataset   = Dataset.from_list(dev_data)
    test_dataset  = Dataset.from_list(test_data)

    print(f"  train sentences : {len(train_dataset)}")
    print(f"  dev   sentences : {len(dev_dataset)}")
    print(f"  test  sentences : {len(test_dataset)}")

    # ── tokeniser ─────────────────────────────────────────────────────────────
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples["tokens"],
            truncation=True,
            max_length=128,
            is_split_into_words=True,
        )
        labels = []
        for i, label in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            prev_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != prev_word_idx:
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)
                prev_word_idx = word_idx
            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    print("Tokenizing...")
    remove_cols = ["tokens", "ner_tags"]
    tokenized_train = train_dataset.map(tokenize_and_align_labels, batched=True, remove_columns=remove_cols)
    tokenized_dev   = dev_dataset.map(tokenize_and_align_labels, batched=True, remove_columns=remove_cols)
    tokenized_test  = test_dataset.map(tokenize_and_align_labels, batched=True, remove_columns=remove_cols)
    print("Tokenization done")

    # ── model ─────────────────────────────────────────────────────────────────
    print("Loading model...")
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_CHECKPOINT,
        num_labels=len(LABEL_LIST),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)

    cuda_avail = torch.cuda.is_available()
    mps_avail = torch.backends.mps.is_available()

    def device_training_plan():
        """Prefer CUDA; default Apple Silicon to CPU unless --mps."""
        if args.cpu:
            return (
                True,
                TRAIN_BATCH_SIZE,
                EVAL_BATCH_SIZE,
                1,
                False,
                "CPU (forced)",
            )
        if cuda_avail:
            return (
                False,
                TRAIN_BATCH_SIZE,
                EVAL_BATCH_SIZE,
                1,
                False,
                "CUDA",
            )
        if mps_avail:
            if not args.mps:
                return (
                    True,
                    TRAIN_BATCH_SIZE,
                    EVAL_BATCH_SIZE,
                    1,
                    False,
                    "CPU (Apple Silicon default — pass --mps to try GPU)",
                )
            micro = max(1, min(TRAIN_BATCH_SIZE, 4))
            accum = max(1, (TRAIN_BATCH_SIZE + micro - 1) // micro)
            eval_bs = max(1, min(EVAL_BATCH_SIZE, 8))
            return False, micro, eval_bs, accum, True, "MPS (micro-batch + checkpoint)"
        return (
            True,
            TRAIN_BATCH_SIZE,
            EVAL_BATCH_SIZE,
            1,
            False,
            "CPU",
        )

    use_cpu, train_bs, eval_bs, grad_accum, grad_ckpt, plan_desc = device_training_plan()

    def make_trainer(m):
        if grad_ckpt:
            m.gradient_checkpointing_enable()
        elif hasattr(m, "gradient_checkpointing_disable"):
            m.gradient_checkpointing_disable()
        targs = TrainingArguments(
            output_dir=OUTPUT_DIR,
            learning_rate=LEARNING_RATE,
            per_device_train_batch_size=train_bs,
            per_device_eval_batch_size=eval_bs,
            gradient_accumulation_steps=grad_accum,
            gradient_checkpointing=grad_ckpt,
            num_train_epochs=NUM_EPOCHS,
            weight_decay=WEIGHT_DECAY,
            eval_strategy="epoch",
            logging_strategy="steps",
            logging_steps=50,
            save_strategy="no",
            report_to="none",
            fp16=False,
            use_cpu=use_cpu,
            dataloader_pin_memory=cuda_avail,
            dataloader_num_workers=0,
        )
        return Trainer(
            model=m,
            args=targs,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_dev,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

    trainer = make_trainer(model)
    print(
        f"Training plan: {plan_desc} | train_bs={train_bs} "
        f"accum={grad_accum} eval_bs={eval_bs} grad_ckpt={grad_ckpt}",
        flush=True,
    )

    print("Starting training...")
    try:
        trainer.train()
    except RuntimeError as e:
        msg = str(e).lower()
        oom_mps = (
            ("out of memory" in msg or "mps backend" in msg)
            and mps_avail
            and not use_cpu
            and not cuda_avail
        )
        if not oom_mps:
            raise
        print(
            "MPS ran out of memory — reloading model and continuing on CPU.",
            flush=True,
        )
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        del trainer
        model = AutoModelForTokenClassification.from_pretrained(
            MODEL_CHECKPOINT,
            num_labels=len(LABEL_LIST),
            id2label=ID2LABEL,
            label2id=LABEL2ID,
        )
        use_cpu = True
        train_bs, eval_bs = TRAIN_BATCH_SIZE, EVAL_BATCH_SIZE
        grad_accum = 1
        grad_ckpt = False
        plan_desc = "CPU (fallback after MPS OOM)"
        trainer = make_trainer(model)
        print(
            f"Training plan: {plan_desc} | train_bs={train_bs} "
            f"accum={grad_accum} eval_bs={eval_bs} grad_ckpt={grad_ckpt}",
            flush=True,
        )
        trainer.train()

    # ── dev: one predict for metrics and IOB2 output ─────────────────────────
    print("\nEvaluating on dev set...")
    dev_predictions, dev_labels, _ = trainer.predict(tokenized_dev)
    results = compute_metrics((dev_predictions, dev_labels))

    print("\n── BASELINE RESULTS (dev) ──")
    for k, v in results.items():
        print(f"  {k:12s}: {v:.4f}")

    print(f"\nSaving dev predictions to {OUTPUT_FILE} ...")
    write_iob2_with_predictions(
        DEV_FILE, dev_data, dev_predictions, OUTPUT_FILE, tokenizer, LABEL_LIST
    )
    print(f"Predictions saved to {OUTPUT_FILE}")
    print("\nTo evaluate run:")
    print(f"  python span_f1.py {DEV_FILE} {OUTPUT_FILE}")

    # Test set: labels are masked (all O); report logits file only, not F1
    print("\nTest set has masked NER labels; skipping F1 on test.")
    test_predictions, _, _ = trainer.predict(tokenized_test)
    print(f"Saving test predictions to {TEST_OUTPUT_FILE} ...")
    write_iob2_with_predictions(
        TEST_FILE, test_data, test_predictions, TEST_OUTPUT_FILE, tokenizer, LABEL_LIST
    )
    print(f"Test predictions saved to {TEST_OUTPUT_FILE}")


if __name__ == "__main__":
    main()
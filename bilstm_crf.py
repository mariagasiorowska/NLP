"""
bilstm_crf.py
-------------
BiLSTM-CRF model for NER on EWT IOB2 data.
No pretrained weights — learns purely from the training split provided.

Usage:
    # standard split (baseline comparison)
    python bilstm_crf.py

    # custom split
    python bilstm_crf.py --train splits/entity_disjoint/train.iob2 \
                         --dev   splits/entity_disjoint/dev.iob2   \
                         --out   predictions_bilstm_ed.iob2

Architecture:
    Embedding → BiLSTM → Linear → CRF

The CRF layer enforces valid IOB2 tag sequences at decode time,
which is standard practice for sequence labelling tasks.
"""

import argparse
import random
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from seqeval.metrics import precision_score, recall_score, f1_score

# ── reproducibility ───────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── config ────────────────────────────────────────────────────────────────────
TRAIN_FILE   = "en_ewt-ud-train.iob2"
DEV_FILE     = "en_ewt-ud-dev.iob2"
OUTPUT_FILE  = "predictions_bilstm.iob2"

# model hyperparameters
EMBEDDING_DIM = 128
HIDDEN_DIM    = 256      # per direction; BiLSTM total = 512
NUM_LAYERS    = 2
DROPOUT       = 0.3
MIN_FREQ      = 1        # minimum token frequency to keep in vocab

# training hyperparameters
LEARNING_RATE = 1e-3
BATCH_SIZE    = 32
NUM_EPOCHS    = 15
GRAD_CLIP     = 5.0

# label set — EWT only uses these 7 labels
LABEL_LIST = ["O", "B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG"]
LABEL2ID   = {l: i for i, l in enumerate(LABEL_LIST)}
ID2LABEL   = {i: l for i, l in enumerate(LABEL_LIST)}
NUM_LABELS = len(LABEL_LIST)

# special tokens
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
PAD_IDX   = 0
UNK_IDX   = 1

# device
DEVICE = (
    torch.device("mps")  if torch.backends.mps.is_available() else
    torch.device("cuda") if torch.cuda.is_available()         else
    torch.device("cpu")
)


# ── data loading ──────────────────────────────────────────────────────────────
def read_ewt(path):
    """
    Reads an EWT .iob2 file.
    Returns list of (tokens, labels) tuples.
    Columns (0-based): 0=index, 1=token, 2=label.
    """
    sentences = []
    tokens, labels = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("#"):
                continue
            if line.strip() == "":
                if tokens:
                    sentences.append((tokens, labels))
                    tokens, labels = [], []
                continue
            parts  = line.split("\t")
            tokens.append(parts[1])
            labels.append(LABEL2ID.get(parts[2], LABEL2ID["O"]))
    if tokens:
        sentences.append((tokens, labels))
    return sentences


# ── vocabulary ────────────────────────────────────────────────────────────────
def build_vocab(sentences, min_freq=MIN_FREQ):
    """Builds word → index vocabulary from training sentences."""
    counts = Counter(tok for toks, _ in sentences for tok in toks)
    vocab  = {PAD_TOKEN: PAD_IDX, UNK_TOKEN: UNK_IDX}
    for word, freq in counts.items():
        if freq >= min_freq:
            vocab[word] = len(vocab)
    return vocab


# ── dataset ───────────────────────────────────────────────────────────────────
class NERDataset(Dataset):
    def __init__(self, sentences, vocab):
        self.data  = sentences
        self.vocab = vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens, labels = self.data[idx]
        token_ids = [
            self.vocab.get(tok, UNK_IDX) for tok in tokens
        ]
        return token_ids, labels


def collate_fn(batch):
    """
    Pads sequences in a batch to the same length.
    Returns:
        token_ids : (batch, max_len)   LongTensor
        labels    : (batch, max_len)   LongTensor  (-100 for padding)
        lengths   : (batch,)           LongTensor
    """
    token_seqs, label_seqs = zip(*batch)
    lengths    = torch.tensor([len(s) for s in token_seqs], dtype=torch.long)
    max_len    = lengths.max().item()

    padded_tokens = torch.zeros(len(batch), max_len, dtype=torch.long)
    padded_labels = torch.full((len(batch), max_len), -100, dtype=torch.long)

    for i, (toks, labs) in enumerate(zip(token_seqs, label_seqs)):
        l = len(toks)
        padded_tokens[i, :l] = torch.tensor(toks, dtype=torch.long)
        padded_labels[i, :l] = torch.tensor(labs, dtype=torch.long)

    return padded_tokens, padded_labels, lengths


# ── CRF layer ─────────────────────────────────────────────────────────────────
class CRF(nn.Module):
    """
    Linear-chain CRF.
    Learns transition scores between labels and enforces
    valid IOB2 sequences at decode time (Viterbi).
    """

    def __init__(self, num_tags):
        super().__init__()
        self.num_tags    = num_tags
        # transition[i, j] = score of transitioning FROM tag j TO tag i
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))
        # enforce: nothing transitions TO the start; FROM the end is fine
        self._init_constraints()

    def _init_constraints(self):
        """
        Hard constraints for valid IOB2:
          - I-X cannot follow O or a different I-Y / B-Y
        We implement soft constraints via large negative init values;
        the model will learn to avoid these during training.
        """
        with torch.no_grad():
            for i, label_i in ID2LABEL.items():
                for j, label_j in ID2LABEL.items():
                    # I-X can only follow B-X or I-X
                    if label_i.startswith("I-"):
                        entity = label_i[2:]
                        if not (label_j in (f"B-{entity}", f"I-{entity}")):
                            self.transitions[i, j] = -10000.0

    def _score_sentence(self, emissions, tags, mask):
        """
        Computes the score of the gold tag sequence.
        emissions : (seq_len, batch, num_tags)
        tags      : (seq_len, batch)
        mask      : (seq_len, batch)  bool
        """
        seq_len, batch = tags.shape
        score = torch.zeros(batch, device=emissions.device)

        for t in range(seq_len):
            emit_score = emissions[t].gather(
                1, tags[t].unsqueeze(1)
            ).squeeze(1)
            score += emit_score * mask[t].float()
            if t > 0:
                trans_score = self.transitions[
                    tags[t], tags[t - 1]
                ]
                score += trans_score * mask[t].float()

        return score

    def _forward_alg(self, emissions, mask):
        """
        Log-partition function via forward algorithm.
        emissions : (seq_len, batch, num_tags)
        mask      : (seq_len, batch)  bool

        Transitions are clamped to [-100, 100] to prevent the -10000
        hard IOB2 constraints from causing numerical overflow (which
        produces negative loss values and stops learning after epoch 1).
        """
        seq_len, batch, num_tags = emissions.shape
        # clamp transitions once per forward pass
        trans = self.transitions.clamp(-100.0, 100.0).unsqueeze(0)

        # initialise with first emission scores
        alpha = emissions[0]   # (batch, num_tags)

        for t in range(1, seq_len):
            # (batch, 1, num_tags) + (1, num_tags, num_tags)
            #   + (batch, num_tags, 1)  → (batch, num_tags, num_tags)
            scores = (alpha.unsqueeze(1)
                      + trans
                      + emissions[t].unsqueeze(2))
            new_alpha = torch.logsumexp(scores, dim=2)
            # keep previous alpha for padded positions
            alpha = torch.where(
                mask[t].unsqueeze(1), new_alpha, alpha
            )

        return torch.logsumexp(alpha, dim=1)   # (batch,)

    def neg_log_likelihood(self, emissions, tags, mask):
        """
        CRF loss = log Z - score(gold).
        emissions : (batch, seq_len, num_tags)
        tags      : (batch, seq_len)
        mask      : (batch, seq_len)  bool
        """
        # transpose to (seq_len, batch, ...)
        emissions = emissions.transpose(0, 1)
        tags      = tags.transpose(0, 1)
        mask      = mask.transpose(0, 1)

        gold_score = self._score_sentence(emissions, tags, mask)
        forward_score = self._forward_alg(emissions, mask)
        return (forward_score - gold_score).mean()

    def decode(self, emissions, mask):
        """
        Viterbi decode. Returns list of list of tag indices.
        emissions : (batch, seq_len, num_tags)
        mask      : (batch, seq_len)  bool
        """
        emissions = emissions.transpose(0, 1)   # (seq_len, batch, num_tags)
        mask      = mask.transpose(0, 1)
        seq_len, batch, num_tags = emissions.shape

        viterbi  = emissions[0]                 # (batch, num_tags)
        backpointers = []

        for t in range(1, seq_len):
            # (batch, num_tags, num_tags)
            scores = viterbi.unsqueeze(1) + self.transitions.unsqueeze(0)
            best_scores, best_tags = scores.max(dim=2)
            new_viterbi = best_scores + emissions[t]
            viterbi = torch.where(
                mask[t].unsqueeze(1), new_viterbi, viterbi
            )
            backpointers.append(best_tags)

        # backtrack
        best_last = viterbi.argmax(dim=1)       # (batch,)
        all_tags  = [best_last]
        for bp in reversed(backpointers):
            best_last = bp.gather(1, best_last.unsqueeze(1)).squeeze(1)
            all_tags.append(best_last)
        all_tags.reverse()                      # (seq_len, batch)

        # convert to list of lists, respecting mask
        lengths = mask.sum(dim=0).tolist()
        result  = []
        for b in range(batch):
            result.append([all_tags[t][b].item()
                           for t in range(int(lengths[b]))])
        return result


# ── model ─────────────────────────────────────────────────────────────────────
class BiLSTMCRF(nn.Module):
    """
    BiLSTM-CRF NER model.
      Embedding → Dropout → BiLSTM → Dropout → Linear → CRF
    """

    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(
            vocab_size, EMBEDDING_DIM, padding_idx=PAD_IDX
        )
        self.lstm = nn.LSTM(
            input_size=EMBEDDING_DIM,
            hidden_size=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=DROPOUT if NUM_LAYERS > 1 else 0.0,
        )
        self.dropout  = nn.Dropout(DROPOUT)
        self.linear   = nn.Linear(HIDDEN_DIM * 2, NUM_LABELS)
        self.crf      = CRF(NUM_LABELS)

    def _get_emissions(self, token_ids, lengths):
        emb  = self.dropout(self.embedding(token_ids))
        # pack for efficiency with variable-length sequences
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        out    = self.dropout(out)
        return self.linear(out)                 # (batch, seq_len, num_tags)

    def forward(self, token_ids, labels, lengths):
        """Returns CRF negative log-likelihood loss."""
        emissions = self._get_emissions(token_ids, lengths)
        mask      = (labels != -100)
        # replace -100 (padding) with 0 for CRF (masked out anyway)
        safe_labels = labels.clone()
        safe_labels[safe_labels == -100] = 0
        return self.crf.neg_log_likelihood(emissions, safe_labels, mask)

    def predict(self, token_ids, lengths):
        """Returns list of predicted tag-id sequences (one per sentence)."""
        emissions = self._get_emissions(token_ids, lengths)
        mask      = torch.zeros(
            token_ids.shape, dtype=torch.bool, device=token_ids.device
        )
        for i, l in enumerate(lengths):
            mask[i, :l] = True
        return self.crf.decode(emissions, mask)


# ── training loop ─────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimiser):
    model.train()
    total_loss = 0.0
    for token_ids, labels, lengths in loader:
        token_ids = token_ids.to(DEVICE)
        labels    = labels.to(DEVICE)
        lengths   = lengths.to(DEVICE)
        optimiser.zero_grad()
        loss = model(token_ids, labels, lengths)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimiser.step()
        total_loss += loss.item()
    return total_loss / len(loader)


# ── evaluation ────────────────────────────────────────────────────────────────
def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for token_ids, labels, lengths in loader:
            token_ids = token_ids.to(DEVICE)
            lengths   = lengths.to(DEVICE)
            preds     = model.predict(token_ids, lengths)
            for pred, lab, length in zip(preds, labels, lengths):
                true_lab = [
                    ID2LABEL[l.item()]
                    for l in lab[:length]
                    if l.item() != -100
                ]
                pred_lab = [ID2LABEL[p] for p in pred]
                all_preds.append(pred_lab)
                all_labels.append(true_lab)
    return {
        "precision": precision_score(all_labels, all_preds),
        "recall":    recall_score(all_labels, all_preds),
        "f1":        f1_score(all_labels, all_preds),
    }


# ── prediction writer ─────────────────────────────────────────────────────────
def save_predictions(model, dev_sentences, dev_raw_path, output_path):
    """
    Writes predictions in EWT IOB2 format (tab-separated, label in col 2).
    Preserves all comment lines and original column structure.
    """
    model.eval()
    vocab      = model._vocab          # attached after building
    pred_labels = []

    with torch.no_grad():
        for tokens, _ in dev_sentences:
            ids     = torch.tensor(
                [vocab.get(t, UNK_IDX) for t in tokens],
                dtype=torch.long
            ).unsqueeze(0).to(DEVICE)
            lengths = torch.tensor([len(tokens)], dtype=torch.long)
            preds   = model.predict(ids, lengths)[0]
            pred_labels.append([ID2LABEL[p] for p in preds])

    with open(dev_raw_path, encoding="utf-8") as f_in, \
         open(output_path, "w", encoding="utf-8") as f_out:

        sent_idx  = 0
        token_idx = 0

        for line in f_in:
            line = line.rstrip("\n")
            if line.startswith("#"):
                f_out.write(line + "\n")
                continue
            if line.strip() == "":
                f_out.write("\n")
                sent_idx  += 1
                token_idx  = 0
                continue
            parts = line.split("\t")
            if sent_idx < len(pred_labels) and \
               token_idx < len(pred_labels[sent_idx]):
                parts[2] = pred_labels[sent_idx][token_idx]
            token_idx += 1
            f_out.write("\t".join(parts) + "\n")

    print(f"  Predictions saved → {output_path}")


# ── main ──────────────────────────────────────────────────────────────────────
def main(train_file, dev_file, output_file):
    print(f"Device : {DEVICE}")
    print(f"Train  : {train_file}")
    print(f"Dev    : {dev_file}")
    print(f"Output : {output_file}\n")

    # ── load data ─────────────────────────────────────────────────────────────
    print("Loading data...")
    train_sentences = read_ewt(train_file)
    dev_sentences   = read_ewt(dev_file)
    print(f"  train: {len(train_sentences):,}  dev: {len(dev_sentences):,}")

    # ── vocabulary ────────────────────────────────────────────────────────────
    vocab = build_vocab(train_sentences)
    print(f"  vocabulary size: {len(vocab):,}")

    # ── datasets & loaders ───────────────────────────────────────────────────
    train_dataset = NERDataset(train_sentences, vocab)
    dev_dataset   = NERDataset(dev_sentences,   vocab)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE,
        shuffle=True,  collate_fn=collate_fn
    )
    dev_loader = DataLoader(
        dev_dataset, batch_size=BATCH_SIZE,
        shuffle=False, collate_fn=collate_fn
    )

    # ── model ─────────────────────────────────────────────────────────────────
    model = BiLSTMCRF(vocab_size=len(vocab)).to(DEVICE)
    model._vocab = vocab          # attach vocab for save_predictions
    optimiser    = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler    = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="max", factor=0.5, patience=2
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  model parameters: {total_params:,}\n")

    # ── training ──────────────────────────────────────────────────────────────
    print(f"Training for {NUM_EPOCHS} epochs...")
    print(f"  {'Epoch':>6}  {'Loss':>8}  {'P':>7}  {'R':>7}  {'F1':>7}")
    print(f"  {'-'*42}")

    best_f1    = 0.0
    best_state = None

    for epoch in range(1, NUM_EPOCHS + 1):
        loss    = train_epoch(model, train_loader, optimiser)
        metrics = evaluate(model, dev_loader)
        f1      = metrics["f1"]
        scheduler.step(f1)

        print(f"  {epoch:>6}  {loss:>8.4f}  "
              f"{metrics['precision']:>7.4f}  "
              f"{metrics['recall']:>7.4f}  "
              f"{f1:>7.4f}")

        if f1 > best_f1:
            best_f1    = f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    # ── final evaluation ──────────────────────────────────────────────────────
    print(f"\n── Best dev F1: {best_f1:.4f} ──")
    model.load_state_dict(best_state)

    final = evaluate(model, dev_loader)
    print("\n── BILSTM-CRF FINAL RESULTS ──")
    for k, v in final.items():
        print(f"  {k:12s}: {v:.4f}")

    # ── save predictions ──────────────────────────────────────────────────────
    print("\nSaving predictions...")
    save_predictions(model, dev_sentences, dev_file, output_file)
    print(f"\nTo evaluate:")
    print(f"  python span_f1.py {dev_file} {output_file}")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BiLSTM-CRF NER on EWT")
    parser.add_argument("--train", default=TRAIN_FILE)
    parser.add_argument("--dev",   default=DEV_FILE)
    parser.add_argument("--out",   default=OUTPUT_FILE)
    args = parser.parse_args()
    main(args.train, args.dev, args.out)

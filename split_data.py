"""
split_data.py
-------------
Creates four alternative train/dev splits from the EWT data:

  1. entity_disjoint  — PER/ORG/LOC (form, type) pairs in train and dev are
                        mutually exclusive; sentences that cannot be placed
                        without breaking that rule go to *excluded* (reported);
                        we still target ~15% of sentences in dev.
  2. frequency_adv    — train on PURELY frequent entities (all 6+),
                        test on PURELY rare entities (all <6), mixed excluded
  3. context_shift    — spaCy on **gold** token boundaries (pre-tokenized Doc);
                        depth + clause count + entity density; train=simple,
                        dev=complex; middle third excluded
  4. cross_domain     — train on reviews+answers+email (+ any unknown),
                        test on newsgroup+weblog

Output structure:
  splits/
  ├── entity_disjoint/  train.iob2  dev.iob2
  ├── frequency_adv/   train.iob2  dev.iob2
  ├── context_shift/  train.iob2  dev.iob2
  └── cross_domain/   train.iob2  dev.iob2

Run:
    pip install spacy
    python -m spacy download en_core_web_sm
    python split_data.py
"""

import os
import random
import statistics
from collections import Counter

random.seed(42)

# ── config ────────────────────────────────────────────────────────────────────
TRAIN_FILE     = "en_ewt-ud-train.iob2"
DEV_FILE       = "en_ewt-ud-dev.iob2"
OUT_DIR        = "splits"

FREQ_THRESHOLD = 6
DISJOINT_TYPES = {"PER", "ORG", "LOC"}

TRAIN_DOMAINS  = {"reviews", "answers", "email"}
DEV_DOMAINS    = {"newsgroup", "weblog"}


# ── sentence class ────────────────────────────────────────────────────────────
class Sentence:
    def __init__(self, comment_lines, token_lines):
        self.comment_lines = comment_lines
        self.token_lines   = token_lines
        self._domain       = None

    @property
    def tokens(self):
        return [l.split("\t")[1] for l in self.token_lines]

    @property
    def labels(self):
        return [l.split("\t")[2] for l in self.token_lines]

    @property
    def length(self):
        return len(self.token_lines)

    @property
    def domain(self):
        if self._domain is None:
            for c in self.comment_lines:
                if c.startswith("# sent_id"):
                    self._domain = c.split("# sent_id = ")[1].split("-")[0]
                    break
            if self._domain is None:
                self._domain = "unknown"
        return self._domain

    def entities(self):
        result = []
        i, labs, toks = 0, self.labels, self.tokens
        while i < len(labs):
            if labs[i].startswith("B-"):
                etype = labs[i][2:]
                span  = [toks[i]]
                j     = i + 1
                while j < len(labs) and labs[j] == f"I-{etype}":
                    span.append(toks[j])
                    j += 1
                result.append((" ".join(span), etype))
                i = j
            else:
                i += 1
        return result


# ── file I/O ──────────────────────────────────────────────────────────────────
def read_sentences(path):
    sentences, comments, tokens = [], [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("#"):
                comments.append(line)
            elif line.strip() == "":
                if tokens:
                    sentences.append(Sentence(comments, tokens))
                    comments, tokens = [], []
            else:
                tokens.append(line)
    if tokens:
        sentences.append(Sentence(comments, tokens))
    return sentences


def write_sentences(sentences, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for sent in sentences:
            for c in sent.comment_lines:
                f.write(c + "\n")
            for t in sent.token_lines:
                f.write(t + "\n")
            f.write("\n")
    print(f"    wrote {len(sentences):,} sentences → {path}")


# ── shared helpers ────────────────────────────────────────────────────────────
def print_entity_stats(train, dev):
    tr = Counter(e for s in train for _, e in s.entities())
    dv = Counter(e for s in dev   for _, e in s.entities())
    print(f"  {'Type':<6} {'Train entities':>16} {'Dev entities':>14}")
    print(f"  {'-'*40}")
    for t in sorted(set(tr) | set(dv)):
        print(f"  {t:<6} {tr[t]:>16,} {dv[t]:>14,}")


def section(title):
    print(f"\n{'='*60}\n  {title}\n{'='*60}")


# ══════════════════════════════════════════════════════════════════════════════
#  SPLIT 1 — ENTITY DISJOINT
# ══════════════════════════════════════════════════════════════════════════════
def make_entity_disjoint_split(all_sents):
    section("Split 1: Entity-Disjoint  (PER + ORG + LOC)")
    print("""
  Strategy: (form, type) pairs in train and in dev are MUTUALLY EXCLUSIVE
  (no (f, type) may appear in both). Dev is filled (up to ~15% of sentences)
  with entity-bearing sentences that do not reuse any (f, type) already
  in train. Remaining entity sentences go to train only if they use no
  (f, type) seen in dev; otherwise they are *excluded* (not written).
  O-only sentences always go to train. PER/ORG/LOC in analysis: most
  separable by surface form.
    """)

    form_freq = Counter()
    for sent in all_sents:
        for form, etype in sent.entities():
            if etype in DISJOINT_TYPES:
                form_freq[(form, etype)] += 1

    target_dev = int(0.15 * len(all_sents))

    def rarity(sent):
        ents = [(f, e) for f, e in sent.entities() if e in DISJOINT_TYPES]
        if not ents:
            return float("inf")
        return min(form_freq[(f, e)] for f, e in ents)

    shuffled = all_sents[:]
    random.shuffle(shuffled)
    shuffled.sort(key=rarity)

    dev_forms, train_forms = set(), set()
    dev_sents, train_sents, excluded_sents = [], [], []

    for sent in shuffled:
        ents = [(f, e) for f, e in sent.entities() if e in DISJOINT_TYPES]
        sent_forms = {(f, e) for f, e in ents}

        if not sent_forms:
            train_sents.append(sent)
            continue

        can_dev = (
            len(dev_sents) < target_dev
            and not (sent_forms & train_forms)
        )
        if can_dev:
            dev_sents.append(sent)
            dev_forms.update(sent_forms)
            continue

        if not (sent_forms & dev_forms):
            train_sents.append(sent)
            train_forms.update(sent_forms)
        else:
            excluded_sents.append(sent)

    print(f"  Total: {len(all_sents):,}  |  "
          f"Train: {len(train_sents):,}  |  Dev: {len(dev_sents):,}  |  "
          f"Excluded: {len(excluded_sents):,}")
    print()
    print("  (Under mutual exclusivity, train vs dev form overlap is 0 by construction.)")
    print()
    print(f"  {'Type':<6} {'Train forms':>12} {'Dev forms':>10} "
          f"{'Overlap':>9} {'Disjoint %':>11}")
    print(f"  {'-'*52}")
    for etype in sorted(DISJOINT_TYPES):
        tr = {f for f, e in train_forms if e == etype}
        dv = {f for f, e in dev_forms   if e == etype}
        ov = tr & dv
        pct = 100 * (1 - len(ov) / len(dv)) if dv else 0.0
        print(f"  {etype:<6} {len(tr):>12,} {len(dv):>10,} "
              f"{len(ov):>9,} {pct:>10.1f}%")
    print()
    print_entity_stats(train_sents, dev_sents)
    return train_sents, dev_sents


# ══════════════════════════════════════════════════════════════════════════════
#  SPLIT 2 — FREQUENCY ADVERSARIAL
# ══════════════════════════════════════════════════════════════════════════════
def make_frequency_adversarial_split(all_sents):
    section("Split 2: Frequency-Adversarial  (pure frequent vs pure rare)")
    print(f"""
  Strategy (pure — no mixed sentences):
    TRAIN    : ALL entities in sentence appear {FREQ_THRESHOLD}+ times globally
    DEV      : ALL entities in sentence appear < {FREQ_THRESHOLD} times globally
    EXCLUDED : sentences with a mix of frequent and rare entities

  Threshold={FREQ_THRESHOLD} from analysis.py. Only 5-9% of surface forms
  appear 6+ times, giving a sharp frequent/rare boundary.
    """)

    form_freq = Counter()
    for sent in all_sents:
        for form, _ in sent.entities():
            form_freq[form] += 1

    train_sents, dev_sents, excluded = [], [], []

    for sent in all_sents:
        ents = sent.entities()
        if not ents:
            train_sents.append(sent)
            continue
        all_freq = all(form_freq[f] >= FREQ_THRESHOLD for f, _ in ents)
        all_rare = all(form_freq[f] <  FREQ_THRESHOLD for f, _ in ents)
        if all_freq:
            train_sents.append(sent)
        elif all_rare:
            dev_sents.append(sent)
        else:
            excluded.append(sent)

    print(f"  Train (pure frequent) : {len(train_sents):,}")
    print(f"  Dev   (pure rare)     : {len(dev_sents):,}")
    print(f"  Excluded (mixed)      : {len(excluded):,}")

    dev_freqs = [form_freq[f] for s in dev_sents for f, _ in s.entities()]
    tr_freqs  = [form_freq[f] for s in train_sents for f, _ in s.entities()]
    if dev_freqs:
        print(f"\n  Dev   freq — mean: {statistics.mean(dev_freqs):.2f}  "
              f"max: {max(dev_freqs)}")
    if tr_freqs:
        print(f"  Train freq — mean: {statistics.mean(tr_freqs):.2f}  "
              f"min: {min(tr_freqs)}")
    print()
    print_entity_stats(train_sents, dev_sents)
    return train_sents, dev_sents


# ══════════════════════════════════════════════════════════════════════════════
#  SPLIT 3 — CONTEXT SHIFT (spaCy)
# ══════════════════════════════════════════════════════════════════════════════
def compute_complexity(sentences):
    """
    Scores each sentence with a composite syntactic complexity measure.
    Uses spaCy en_core_web_sm (dependency parse) on a pre-tokenized Doc
    (same token boundaries as EWT; spaces between tokens are assumed).
    Falls back to sentence length if spaCy is unavailable.

    Composite score (normalised 0-1):
      0.5 * dependency tree depth   — primary syntactic signal
      0.3 * clause count            — subordination / embedding
      0.2 * entity density          — NER difficulty per token
    """
    try:
        import spacy
        from spacy.tokens import Doc
        nlp = spacy.load("en_core_web_sm")
        use_spacy = True
        print("  spaCy loaded — using dependency-based complexity (gold token boundaries via Doc)")
    except (ImportError, OSError):
        print("  WARNING: spaCy not available — falling back to sentence length")
        print("  To use full complexity scoring:")
        print("    pip install spacy")
        print("    python -m spacy download en_core_web_sm")
        use_spacy = False

    raw = {}

    if use_spacy:
        def tree_depth(token):
            d = 0
            while token.head != token:
                token = token.head
                d += 1
            return d

        # English: relative/adnominal "acl:relcl" is used in UD; keep "relcl" too.
        clause_deps = {
            "ROOT", "ccomp", "advcl", "relcl", "acl:relcl",
        }

        for idx, sent in enumerate(sentences):
            words  = list(sent.tokens)
            n      = len(words)
            spaces = [True] * (n - 1) + [False] if n else []
            doc    = Doc(nlp.vocab, words=words, spaces=spaces)
            for _, proc in nlp.pipeline:
                doc = proc(doc)
            depth = max((tree_depth(t) for t in doc), default=0)
            clause_count = sum(1 for t in doc if t.dep_ in clause_deps)
            ent_density  = len(sent.entities()) / max(sent.length, 1)
            raw[idx]     = (depth, clause_count, ent_density)
    else:
        for idx, sent in enumerate(sentences):
            raw[idx] = (sent.length, 0, 0)

    depths    = [v[0] for v in raw.values()]
    clauses   = [v[1] for v in raw.values()]
    densities = [v[2] for v in raw.values()]

    def norm(val, vals):
        mn, mx = min(vals), max(vals)
        return (val - mn) / (mx - mn) if mx > mn else 0.0

    return {
        idx: (0.5 * norm(d, depths) +
              0.3 * norm(c, clauses) +
              0.2 * norm(e, densities))
        for idx, (d, c, e) in raw.items()
    }


def make_context_shift_split(all_sents):
    section("Split 3: Context-Shift  (spaCy syntactic complexity)")
    print("""
  Strategy:
    Score every sentence with a composite syntactic complexity measure:
      dependency tree depth (0.5) + clause count (0.3) + entity density (0.2)
    TRAIN    : bottom third — syntactically simplest sentences
    DEV      : top third    — syntactically most complex sentences
    EXCLUDED : middle third — excluded for sharp contrast

  Tool: spaCy en_core_web_sm; parses pre-tokenized Doc(s) (EWT word boundaries),
  not a whitespace-rejoined string.
  This directly addresses the professor's question about which tools
  to use for studying syntactically different contexts.
    """)

    print("  Scoring syntactic complexity (this may take ~1 min)...")
    scores = compute_complexity(all_sents)

    sorted_ids = sorted(scores, key=scores.get)
    n          = len(sorted_ids)
    third      = n // 3

    train_ids = set(sorted_ids[:third])
    dev_ids   = set(sorted_ids[2 * third:])

    train_sents = [s for i, s in enumerate(all_sents) if i in train_ids]
    dev_sents   = [s for i, s in enumerate(all_sents) if i in dev_ids]
    excluded    = n - len(train_sents) - len(dev_sents)

    tr_sc = [scores[i] for i in sorted_ids[:third]]
    dv_sc = [scores[i] for i in sorted_ids[2 * third:]]
    tr_ln = [s.length for s in train_sents]
    dv_ln = [s.length for s in dev_sents]

    print(f"\n  Train (simplest)  : {len(train_sents):,}  "
          f"mean score: {statistics.mean(tr_sc):.3f}  "
          f"mean length: {statistics.mean(tr_ln):.1f}")
    print(f"  Dev   (complex)   : {len(dev_sents):,}  "
          f"mean score: {statistics.mean(dv_sc):.3f}  "
          f"mean length: {statistics.mean(dv_ln):.1f}")
    print(f"  Excluded (middle) : {excluded:,}")
    print()
    print_entity_stats(train_sents, dev_sents)
    return train_sents, dev_sents


# ══════════════════════════════════════════════════════════════════════════════
#  SPLIT 4 — CROSS DOMAIN
# ══════════════════════════════════════════════════════════════════════════════
def make_cross_domain_split(all_sents):
    section("Split 4: Cross-Domain")
    print(f"""
  EWT contains 5 domains (from sent_id prefixes):
    reviews   — product/place reviews     (informal, opinionated)
    answers   — Yahoo Answers             (informal, conversational)
    email     — personal emails           (informal, personal)
    newsgroup — Usenet newsgroups         (semi-formal, longer discussion)
    weblog    — personal blogs            (informal but longer form)

  TRAIN : {sorted(TRAIN_DOMAINS)}
  DEV   : {sorted(DEV_DOMAINS)}

  Rationale: reviews/answers/email share informal register and are
  the largest domains (best training signal). Newsgroup and weblog
  have different stylistic conventions — testing on these measures
  cross-register generalisation. Any sentence with an unlisted domain
  is assigned to TRAIN so no data is dropped.
    """)

    union       = TRAIN_DOMAINS | DEV_DOMAINS
    other_sents = [s for s in all_sents if s.domain not in union]

    train_sents = [s for s in all_sents
                   if s.domain in TRAIN_DOMAINS or s.domain not in union]
    dev_sents   = [s for s in all_sents if s.domain in DEV_DOMAINS]

    domain_counts = Counter(s.domain for s in all_sents)
    print("  Domain distribution:")
    for domain, count in domain_counts.most_common():
        if domain in TRAIN_DOMAINS:
            tag = "TRAIN"
        elif domain in DEV_DOMAINS:
            tag = "DEV  "
        else:
            tag = "TRAIN"  # other → train
        print(f"    {tag}  {domain:<12} {count:>5,} sentences")

    print(f"\n  Train : {len(train_sents):,}")
    print(f"  Dev   : {len(dev_sents):,}")
    if other_sents:
        odoms = ", ".join(sorted({s.domain for s in other_sents}))
        print(
            f"  Other domain(s) (→ TRAIN): {odoms}  "
            f"— {len(other_sents):,} sentence(s)"
        )
    print()
    print_entity_stats(train_sents, dev_sents)
    return train_sents, dev_sents


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("Reading EWT files...")
    train_sents = read_sentences(TRAIN_FILE)
    dev_sents   = read_sentences(DEV_FILE)
    all_sents   = train_sents + dev_sents
    print(f"  Train: {len(train_sents):,}  "
          f"Dev: {len(dev_sents):,}  "
          f"Total: {len(all_sents):,}")

    ed_tr, ed_dv = make_entity_disjoint_split(all_sents)
    write_sentences(ed_tr, os.path.join(OUT_DIR, "entity_disjoint", "train.iob2"))
    write_sentences(ed_dv, os.path.join(OUT_DIR, "entity_disjoint", "dev.iob2"))

    fa_tr, fa_dv = make_frequency_adversarial_split(all_sents)
    write_sentences(fa_tr, os.path.join(OUT_DIR, "frequency_adv", "train.iob2"))
    write_sentences(fa_dv, os.path.join(OUT_DIR, "frequency_adv", "dev.iob2"))

    cs_tr, cs_dv = make_context_shift_split(all_sents)
    write_sentences(cs_tr, os.path.join(OUT_DIR, "context_shift", "train.iob2"))
    write_sentences(cs_dv, os.path.join(OUT_DIR, "context_shift", "dev.iob2"))

    cd_tr, cd_dv = make_cross_domain_split(all_sents)
    write_sentences(cd_tr, os.path.join(OUT_DIR, "cross_domain", "train.iob2"))
    write_sentences(cd_dv, os.path.join(OUT_DIR, "cross_domain", "dev.iob2"))

    section("ALL SPLITS COMPLETE")
    print("""
  splits/
  ├── entity_disjoint/   train.iob2   dev.iob2
  ├── frequency_adv/     train.iob2   dev.iob2
  ├── context_shift/     train.iob2   dev.iob2
  └── cross_domain/      train.iob2   dev.iob2

  Next step: e.g. python3 bilstm_crf.py
    """)


if __name__ == "__main__":
    main()

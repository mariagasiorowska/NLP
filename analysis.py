"""
analysis.py
-----------
Distribution analysis of the EWT dataset.
Answers professor questions:
  1. How is the entity distribution in the standard train/dev/test split?
  2. Which entity types have the most overlap / are most separable?

Run with:
    python analysis.py
"""

from collections import defaultdict, Counter

# ── config ────────────────────────────────────────────────────────────────────
TRAIN_FILE = "en_ewt-ud-train.iob2"
DEV_FILE   = "en_ewt-ud-dev.iob2"
TEST_FILE  = "en_ewt-ud-test-masked.iob2"

# ── file reader ───────────────────────────────────────────────────────────────
def read_ewt(path):
    """
    Returns a list of sentences.
    Each sentence is a list of (token, label) tuples.
    """
    sentences = []
    current = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("#"):
                continue
            if line.strip() == "":
                if current:
                    sentences.append(current)
                    current = []
                continue
            parts = line.split("\t")
            token = parts[1]
            label = parts[2]
            current.append((token, label))
    if current:
        sentences.append(current)
    return sentences


# ── entity span extractor ─────────────────────────────────────────────────────
def extract_entities(sentences):
    """
    Returns:
      entities       : list of (surface_form, entity_type) for every span
      type_to_forms  : dict mapping entity_type -> Counter of surface forms
    """
    entities = []
    type_to_forms = defaultdict(Counter)

    for sentence in sentences:
        i = 0
        while i < len(sentence):
            token, label = sentence[i]
            if label.startswith("B-"):
                etype = label[2:]
                span_tokens = [token]
                j = i + 1
                while j < len(sentence) and sentence[j][1] == f"I-{etype}":
                    span_tokens.append(sentence[j][0])
                    j += 1
                surface = " ".join(span_tokens)
                entities.append((surface, etype))
                type_to_forms[etype][surface] += 1
                i = j
            else:
                i += 1

    return entities, type_to_forms


# ── helpers ───────────────────────────────────────────────────────────────────
def surface_forms(type_to_forms):
    """Flat set of all surface forms across all types."""
    forms = set()
    for counter in type_to_forms.values():
        forms.update(counter.keys())
    return forms


def surface_forms_by_type(type_to_forms, etype):
    return set(type_to_forms[etype].keys())


def section(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    print("Reading EWT files...")
    train_sents = read_ewt(TRAIN_FILE)
    dev_sents   = read_ewt(DEV_FILE)
    test_sents  = read_ewt(TEST_FILE)

    print(f"  Train sentences : {len(train_sents)}")
    print(f"  Dev   sentences : {len(dev_sents)}")
    print(f"  Test  sentences : {len(test_sents)}")

    train_ents, train_forms = extract_entities(train_sents)
    dev_ents,   dev_forms   = extract_entities(dev_sents)
    # test is masked so no gold labels — we skip entity extraction for test

    # ── 1. overall entity counts ──────────────────────────────────────────────
    section("1. OVERALL ENTITY COUNTS")

    all_types = sorted(set(train_forms.keys()) | set(dev_forms.keys()))

    print(f"\n{'Type':<8} {'Train tokens':>14} {'Train unique':>14} "
          f"{'Dev tokens':>12} {'Dev unique':>12}")
    print("-" * 64)
    for etype in all_types:
        tr_tok = sum(train_forms[etype].values())
        tr_uniq = len(train_forms[etype])
        dv_tok = sum(dev_forms[etype].values())
        dv_uniq = len(dev_forms[etype])
        print(f"{etype:<8} {tr_tok:>14,} {tr_uniq:>14,} "
              f"{dv_tok:>12,} {dv_uniq:>12,}")

    total_tr = len(train_ents)
    total_dv = len(dev_ents)
    print(f"\n  Total train entities : {total_tr:,}")
    print(f"  Total dev   entities : {total_dv:,}")

    # ── 2. overlap analysis ───────────────────────────────────────────────────
    section("2. SURFACE FORM OVERLAP (train vs dev)")

    print(f"\n{'Type':<8} {'Train unique':>14} {'Dev unique':>10} "
          f"{'Overlap':>10} {'Overlap %':>10} {'Unseen in dev':>15}")
    print("-" * 72)

    overlap_summary = {}
    for etype in all_types:
        tr_set  = surface_forms_by_type(train_forms, etype)
        dv_set  = surface_forms_by_type(dev_forms,   etype)
        overlap = tr_set & dv_set
        unseen  = dv_set - tr_set
        pct     = 100 * len(overlap) / len(dv_set) if dv_set else 0
        overlap_summary[etype] = {
            "train": tr_set, "dev": dv_set,
            "overlap": overlap, "unseen": unseen
        }
        print(f"{etype:<8} {len(tr_set):>14,} {len(dv_set):>10,} "
              f"{len(overlap):>10,} {pct:>9.1f}% {len(unseen):>15,}")

    # ── 3. frequency distribution ─────────────────────────────────────────────
    section("3. FREQUENCY DISTRIBUTION IN TRAINING SET")

    for etype in all_types:
        counts = train_forms[etype]
        total  = sum(counts.values())
        freq_1 = sum(1 for c in counts.values() if c == 1)
        freq_2_5 = sum(1 for c in counts.values() if 2 <= c <= 5)
        freq_6p  = sum(1 for c in counts.values() if c > 5)

        print(f"\n  {etype}  (total mentions: {total:,}, unique forms: {len(counts):,})")
        print(f"    appear exactly once  : {freq_1:,}  ({100*freq_1/len(counts):.1f}%)")
        print(f"    appear 2–5 times     : {freq_2_5:,}  ({100*freq_2_5/len(counts):.1f}%)")
        print(f"    appear 6+ times      : {freq_6p:,}  ({100*freq_6p/len(counts):.1f}%)")

        print(f"    Top 10 most frequent:")
        for form, cnt in counts.most_common(10):
            print(f"      {cnt:>5}x  {form}")

    # ── 4. unseen entities in dev ─────────────────────────────────────────────
    section("4. UNSEEN ENTITIES IN DEV (never seen during training)")

    for etype in all_types:
        unseen = overlap_summary[etype]["unseen"]
        dv_set = overlap_summary[etype]["dev"]
        print(f"\n  {etype}: {len(unseen):,} unseen out of {len(dv_set):,} "
              f"({100*len(unseen)/len(dv_set):.1f}% of dev entities are NEW)")
        print(f"  Examples of unseen {etype} entities:")
        for form in sorted(unseen)[:15]:
            print(f"    - {form}")

    # ── 5. separability assessment ────────────────────────────────────────────
    section("5. SEPARABILITY ASSESSMENT (for entity-disjoint split)")

    print("""
  This section helps answer: which entity type is best to separate?

  Key insight: to create an entity-disjoint split, we need entity types
  where the same surface form rarely appears in both train and test.
  The higher the 'unseen %', the more naturally disjoint that type already is.
    """)

    for etype in all_types:
        unseen = overlap_summary[etype]["unseen"]
        dv_set = overlap_summary[etype]["dev"]
        pct_unseen = 100 * len(unseen) / len(dv_set) if dv_set else 0
        tr_set = overlap_summary[etype]["train"]
        shared = overlap_summary[etype]["overlap"]

        # how many train sentences contain this entity type?
        train_sentences_with_type = sum(
            1 for s in train_sents
            if any(lbl == f"B-{etype}" for _, lbl in s)
        )
        dev_sentences_with_type = sum(
            1 for s in dev_sents
            if any(lbl == f"B-{etype}" for _, lbl in s)
        )

        print(f"  {etype}:")
        print(f"    Unseen in dev          : {pct_unseen:.1f}%")
        print(f"    Shared surface forms   : {len(shared):,}")
        print(f"    Train sentences with {etype}: {train_sentences_with_type:,}")
        print(f"    Dev   sentences with {etype}: {dev_sentences_with_type:,}")

        if pct_unseen > 40:
            note = "✓ GOOD candidate for entity-disjoint split"
        elif pct_unseen > 20:
            note = "~ MODERATE — partial disjoint possible"
        else:
            note = "✗ HIGH overlap — hard to make disjoint"
        print(f"    Assessment             : {note}")

    # ── 6. context diversity (for context-shift split) ────────────────────────
    section("6. SENTENCE LENGTH DISTRIBUTION (proxy for context complexity)")

    def sent_lengths(sents):
        return [len(s) for s in sents]

    tr_lens = sent_lengths(train_sents)
    dv_lens = sent_lengths(dev_sents)

    import statistics
    print(f"\n  Train — mean length: {statistics.mean(tr_lens):.1f}  "
          f"median: {statistics.median(tr_lens):.1f}  "
          f"max: {max(tr_lens)}")
    print(f"  Dev   — mean length: {statistics.mean(dv_lens):.1f}  "
          f"median: {statistics.median(dv_lens):.1f}  "
          f"max: {max(dv_lens)}")

    # length buckets
    def bucket(lengths, label):
        short  = sum(1 for l in lengths if l <= 10)
        medium = sum(1 for l in lengths if 11 <= l <= 25)
        long_  = sum(1 for l in lengths if l > 25)
        total  = len(lengths)
        print(f"  {label} length buckets:")
        print(f"    short  (≤10 tokens) : {short:,}  ({100*short/total:.1f}%)")
        print(f"    medium (11–25)       : {medium:,}  ({100*medium/total:.1f}%)")
        print(f"    long   (>25 tokens)  : {long_:,}  ({100*long_/total:.1f}%)")

    bucket(tr_lens, "Train")
    bucket(dv_lens, "Dev  ")

    print("\n  → Context-shift split strategy: train on SHORT sentences,")
    print("    test on LONG sentences. This changes syntactic complexity")
    print("    without needing external tools like spaCy.")

    section("DONE — use these numbers in your report")
    print("  Key findings to report:")
    print("  1. Entity counts and type distribution per split")
    print("  2. Overlap % per type → justifies entity-disjoint split focus")
    print("  3. Frequency distribution → justifies frequency-adversarial split")
    print("  4. Sentence length distribution → justifies context-shift split\n")


if __name__ == "__main__":
    main()

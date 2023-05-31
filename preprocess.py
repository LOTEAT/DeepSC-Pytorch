"""
Author: LOTEAT
Date: 2023-05-30 20:12:50
"""
import unicodedata
import re
from w3lib.html import remove_tags
import pickle
import argparse
import os
import json
from tqdm import tqdm

SPECIAL_TOKENS = {
  '<PAD>': 0,
  '<START>': 1,
  '<END>': 2,
  '<UNK>': 3,
}


def helper():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", default="data/europarl/en", type=str)
    parser.add_argument(
        "--out_train_dir", default="data/europarl/train_data.pkl", type=str
    )
    parser.add_argument(
        "--out_test_dir", default="data/europarl/test_data.pkl", type=str
    )
    parser.add_argument("--out_vocab", default="data/europarl/vocab.json", type=str)
    args = parser.parse_args()
    return args


def unicode_to_ascii(s):
    return "".join(
        c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn"
    )


def normalize_string(s):
    # normalize unicode characters
    s = unicode_to_ascii(s)
    # remove the XML-tags
    s = remove_tags(s)
    # add white space before !.?
    s = re.sub(r"([!.?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s)
    # change to lower letter
    s = s.lower()
    return s


def cutted_data(cleaned, MIN_LENGTH=4, MAX_LENGTH=30):
    cutted_lines = list()
    for line in cleaned:
        length = len(line.split())
        if length > MIN_LENGTH and length < MAX_LENGTH:
            line = [word for word in line.split()]
            cutted_lines.append(" ".join(line))
    return cutted_lines


def process(text_path):
    fop = open(text_path, "r", encoding="utf8")
    raw_data = fop.read()
    sentences = raw_data.strip().split("\n")
    raw_data_input = [normalize_string(data) for data in sentences]
    raw_data_input = cutted_data(raw_data_input)
    fop.close()
    return raw_data_input


def tokenize(
    s,
    delim=" ",
    add_start_token=True,
    add_end_token=True,
    punct_to_keep=None,
    punct_to_remove=None,
):
    """
    Tokenize a sequence, converting a string s into a list of (string) tokens by
    splitting on the specified delimiter. Optionally keep or remove certain
    punctuation marks and add start and end tokens.
    """
    if punct_to_keep is not None:
        for p in punct_to_keep:
            s = s.replace(p, "%s%s" % (delim, p))

    if punct_to_remove is not None:
        for p in punct_to_remove:
            s = s.replace(p, "")

    tokens = s.split(delim)
    if add_start_token:
        tokens.insert(0, "<START>")
    if add_end_token:
        tokens.append("<END>")
    return tokens


def build_vocab(
    sequences,
    token_to_idx={},
    min_token_count=1,
    delim=" ",
    punct_to_keep=None,
    punct_to_remove=None,
):
    token_to_count = {}
    for seq in sequences:
        seq_tokens = tokenize(
            seq,
            delim=delim,
            punct_to_keep=punct_to_keep,
            punct_to_remove=punct_to_remove,
            add_start_token=False,
            add_end_token=False,
        )
        for token in seq_tokens:
            if token not in token_to_count:
                token_to_count[token] = 0
            token_to_count[token] += 1

    for token, count in sorted(token_to_count.items()):
        if count >= min_token_count:
            token_to_idx[token] = len(token_to_idx)

    return token_to_idx


def preprocess(args):
    """
    Preprocesses the raw data.
    """
    sentences = []
    print("------------Preprocess Raw Text------------")
    for fn in tqdm(os.listdir(args.in_dir)):
        if fn.endswith(".txt"):
            process_sentences = process(os.path.join(args.in_dir, fn))
            sentences += process_sentences

    # remove the same sentences
    sentences = list(dict.fromkeys(sentences))
    print("Number of sentences: {}".format(len(sentences)))
    print("------------Build Vocabulary------------")
    token_to_idx = build_vocab(
        sentences, SPECIAL_TOKENS, punct_to_keep=[";", ","], punct_to_remove=["?", "."]
    )
    
    vocab = {'token_to_idx': token_to_idx}
    print('Number of words in Vocab: {}'.format(len(token_to_idx)))
    
    
        # save the vocab
    if args.out_vocab != '':
        with open(args.out_vocab, 'w') as f:
            json.dump(vocab, f)

    print('Start encoding txt')
    results = []
    count_len = []
    for seq in tqdm(sentences):
        words = tokenize(seq, punct_to_keep=[';', ','], punct_to_remove=['?', '.'])
        tokens = [token_to_idx[word] for word in words]
        count_len.append(len(tokens))
        results.append(tokens)
        
    print('Writing Data')
    train_data = results[: round(len(results) * 0.9)]
    test_data = results[round(len(results) * 0.9):]

    with open(args.out_train_dir, 'wb') as f:
        pickle.dump(train_data, f)
    with open(args.out_test_dir, 'wb') as f:
        pickle.dump(test_data, f)

if __name__ == '__main__':
    args = helper()
    preprocess(args)


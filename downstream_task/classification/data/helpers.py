import functools
import json
import os
from torch.utils.data import DataLoader
from torchvision import transforms
from .dataset import JsonlDataset
from transformers import BertTokenizer
from torch.utils.data import DataLoader

from data.dataset import JsonlDataset
from data.vocab import Vocab


def get_transforms(args):
    if args.openi:
        return transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
    else:
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
    )


def get_labels_and_frequencies(path):
    label_freqs = Counter()
    data_labels = [json.loads(line)["label"] for line in open(path)]
    if type(data_labels) == list:
        for label_row in data_labels:
            if label_row == '':
                label_row = ["'Others'"]
            else:
                label_row = label_row.split(', ')

            label_freqs.update(label_row)
    else:
        pass
    return list(label_freqs.keys()), label_freqs


def get_glove_words(path):
    word_list = []
    for line in open(path):
        w, _ = line.split(" ", 1)
        word_list.append(w)
    return word_list


def get_vocab(args):
    vocab = Vocab()
    bert_tokenizer = BertTokenizer.from_pretrained(
        args.bert_model, do_lower_case=True
    )
    vocab.stoi = bert_tokenizer.vocab
    vocab.itos = bert_tokenizer.ids_to_tokens
    vocab.vocab_sz = len(vocab.itos)

    return vocab


def collate_fn(batch, args):
    lens = [len(row[0]) for row in batch]
    bsz, max_seq_len = len(batch), max(lens)

    mask_tensor = torch.zeros(bsz, max_seq_len).long()
    text_tensor = torch.zeros(bsz, max_seq_len).long()
    segment_tensor = torch.zeros(bsz, max_seq_len).long()

    img_tensor = None
    img_tensor = torch.stack([row[2] for row in batch])

    if args.task_type == "multilabel":
        # Multilabel case
        tgt_tensor = torch.stack([row[3] for row in batch])
    else:
        # Single Label case
        tgt_tensor = torch.cat([row[3] for row in batch]).long()

    for i_batch, (input_row, length) in enumerate(zip(batch, lens)):
        tokens, segment = input_row[:2]
        text_tensor[i_batch, :length] = tokens
        segment_tensor[i_batch, :length] = segment
        mask_tensor[i_batch, :length] = 1

    return text_tensor, segment_tensor, mask_tensor, img_tensor, tgt_tensor


def get_data_loaders(args):
    train_path = os.path.join(args.data_path, args.Train_dset_name)
    
    all_labels = set()
    with open(train_path) as f:
        for line in f:
            data = json.loads(line)
            if data["label"]:
                labels = data["label"].split(', ') if isinstance(data["label"], str) else data["label"]
                all_labels.update(labels)
    
    args.labels = sorted(list(all_labels))
    args.n_classes = len(args.labels)
    

    tokenizer = BertTokenizer.from_pretrained(args.bert_model)
    vocab = get_vocab(args)  # You'll need to implement this if it doesn't exist
    
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225]),
    ])

    img_path = os.path.join(args.data_path, "images")

    train_dataset = JsonlDataset(
        data_path=os.path.join(args.data_path, args.Train_dset_name),
        tokenizer=tokenizer,
        transforms=transform,  # Changed from transform to transforms
        vocab=vocab,
        args=args,
        img_path=img_path,
        test=False
    )

    val_dataset = JsonlDataset(
        data_path=os.path.join(args.data_path, args.Valid_dset_name),
        tokenizer=tokenizer,
        transforms=transform,
        vocab=vocab,
        args=args,
        img_path=img_path,
        test=True
    )

    # Rest of your DataLoader code remains the same
    train_loader = DataLoader(...)
    val_loader = DataLoader(...)
    
    return train_loader, val_loader
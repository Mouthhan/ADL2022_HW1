import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from tqdm.auto import tqdm

from model import LSTM_Tagger
from dataset import SeqClsDataset
from utils import Vocab
from torch.utils.data import DataLoader
import numpy as np
import random

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]

def same_seeds(seed):
    # Python built-in random module
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Torch
    torch.manual_seed(seed)
    # Cuda
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def main(args):
    same_seeds(63)
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "tag2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text())
            for split, path in data_paths.items()}
    datasets: Dict[str, SeqClsDataset] = {
        split: SeqClsDataset(split_data, vocab, intent2idx, args.max_len)
        for split, split_data in data.items()
    }
    embeddings = torch.load(args.cache_dir / "embeddings.pt")
    BATCH_SIZE = 128
    max_len = 32
    tag_all = [0,0,0,0,0,0,0,0,0]
    longest = 0
    avg = 0
    avg_c = 0

    for dataset in datasets:
        for idx, all_data in enumerate(datasets[dataset]):  # get data
            result = torch.zeros(max_len, 300)  # template input
            result_tags = torch.zeros(max_len, dtype=torch.long)
            result_tags += 4 # 4 in tag2idx is O type
            count = 0  # check length
            input_list = all_data['tokens']
            avg += len(input_list)
            avg_c += 1
            if len(input_list) >longest:
                longest = len(input_list)
            if len(input_list) > 0:
                for word in input_list:
                    if word in vocab.tokens:
                        result[count] = embeddings[vocab.token2idx[word]]
                        count += 1
                        if count >= max_len:
                            break
            all_data['tokens'] = result
            count = 0
            for tag in all_data['tags']:
                result_tags[count] = SeqClsDataset.label2idx(datasets[dataset],
                                                             label=tag)
                tag_all[SeqClsDataset.label2idx(datasets[dataset],
                                                             label=tag)] += 1
                count += 1
                if count >= max_len:
                    break
            all_data['tags'] = result_tags
    train_loader = DataLoader(
        datasets['train'], batch_size=BATCH_SIZE, shuffle=True)
    eval_loader = DataLoader(
        datasets['eval'], batch_size=BATCH_SIZE, shuffle=False)
    device = get_device()
    print(device)
    weight_loss = torch.FloatTensor(tag_all).to(device)
    num_epoch = 50
    input_dim = 300  # glove's dim
    class_num = len(intent2idx)
    hidden = 1024
    layer = 2
    lr = 3e-4
    model = LSTM_Tagger(input_dim, hidden, layer, class_num, max_len).to(device)

    optimizer = torch.optim.AdamW(model.parameters(),lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)

    best_acc = 0.0  # to save model
    model_path = './ckpt/slot/best.ckpt'
    for epoch in range(num_epoch):
        criterion = nn.CrossEntropyLoss()
        train_acc = 0.0
        train_all = 0.0
        train_loss = 0.0
        val_acc = 0.0
        val_all = 0.0
        val_loss = 0.0
        model.train()  # train mode
        for i, data in enumerate(tqdm(train_loader)):
            
            inputs, labels, index = data['tokens'], data['tags'], data['id']
            inputs, labels = inputs.to(
                device), labels.to(device)
            batch = labels.shape[0]
            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = outputs.view(batch * max_len, -1)
            labels = labels.view(-1)
            batch_loss = criterion(outputs, labels)
            _, train_pred = torch.max(outputs, 1)  
            batch_loss.backward()
            optimizer.step()
            scheduler.step()
            train_all += batch
            train_pred= train_pred.view(batch, -1)
            labels = labels.view(batch, -1)
            batch_results = np.array([np.array_equal(train_pred.cpu().numpy()[idx],labels.cpu()[idx]) for idx in range(batch)])
            train_acc += batch_results.sum().item()
            train_loss += batch_loss.item()

        # validation
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(tqdm(eval_loader)):
                inputs, labels, index = data['tokens'], data['tags'], data['id']
                inputs, labels = inputs.to(
                    device), labels.to(device)
                batch = labels.shape[0]
                outputs = model(inputs)
                outputs = outputs.view(batch * max_len, -1)
                labels = labels.view(-1)
                batch_loss = criterion(outputs, labels)
                _, val_pred = torch.max(outputs, 1)
                val_all += batch
                val_pred= val_pred.view(batch, -1)
                labels = labels.view(batch, -1)
                batch_results = np.array([np.array_equal(val_pred.cpu().numpy()[idx],labels.cpu()[idx]) for idx in range(batch)])
                val_acc += batch_results.sum().item()
                val_loss += batch_loss.item()

            print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f} | Val Acc: {:3.6f} loss: {:3.6f}'.format(
                epoch + 1, num_epoch, train_acc/train_all, train_loss/len(
                    train_loader), val_acc/val_all, val_loss/len(eval_loader)
            ))

            # save if improve
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), model_path)
                print('saving model with acc {:.3f}'.format(
                    best_acc/val_all))
            else:
                print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f}'.format(
                    epoch + 1, num_epoch, train_acc /
                    len(train_loader), train_loss/len(train_loader)
                ))


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/slot/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/slot/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/slot/",
    )

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-3)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    parser.add_argument("--num_epoch", type=int, default=2)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)

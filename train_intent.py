import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
from tqdm.auto import tqdm
from tqdm import trange
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import SeqClsDataset
from model import  LSTM
from utils import Vocab
import random
import numpy as np

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

    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    datasets: Dict[str, SeqClsDataset] = {
        split: SeqClsDataset(split_data, vocab, intent2idx, args.max_len)
        for split, split_data in data.items()
    }
    BATCH_SIZE = 128
    max_len = 32
    # read train & dev dataset
    embeddings = torch.load(args.cache_dir / "embeddings.pt")
    avg_len = []
    for dataset in datasets:
        for idx, all_data in enumerate(datasets[dataset]):  # get data
            result = torch.zeros(max_len, 300)  # template input
            count = 0  # check length
            input_list = all_data['text'].split()
            avg_len.append(len(input_list))
            if len(input_list) > 0:
                for word in input_list:
                    if word in vocab.tokens:
                        result[count] = embeddings[vocab.token2idx[word]]
                        count += 1
                        if count >= max_len:
                            break
            all_data['text'] = result
            # all_data['text'] = [embeddings[vocab.encode(word)]
            #                     for word in all_data['text']]
            all_data['intent'] = SeqClsDataset.label2idx(datasets[dataset],
                                                         label=all_data['intent'])  # str to int
    train_loader = DataLoader(
        datasets['train'], batch_size=BATCH_SIZE, shuffle=True)
    eval_loader = DataLoader(
        datasets['eval'], batch_size=BATCH_SIZE, shuffle=False)
    
    device = get_device()
    print(device)
    num_epoch = 30
    input_dim = 300  # glove's dim
    class_num = len(intent2idx)
    print(class_num)
    hidden = 1024
    layer = 2
    model = LSTM(input_dim, hidden, layer, class_num).to(device)

    optimizer = torch.optim.AdamW(model.parameters(),lr = 3e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)

    best_acc = 0.0  # to save model
    model_path = './ckpt/intent/best.ckpt'
    for epoch in range(num_epoch):
        criterion = nn.CrossEntropyLoss()
        train_acc = 0.0
        train_all = 0.0
        train_loss = 0.0
        val_acc = 0.0
        val_all = 0.0
        val_loss = 0.0
        # training
        model.train()  # train mode
        for i, data in enumerate(tqdm(train_loader)):
            inputs, labels, index = data['text'], data['intent'], data['id']
            inputs, labels = inputs.to(
                device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            batch_loss = criterion(outputs, labels)
            _, train_pred = torch.max(outputs, 1)  # get pred label
            batch_loss.backward()
            optimizer.step()
            scheduler.step()
            train_all += labels.shape[0]
            train_acc += (train_pred.cpu() == labels.cpu()).sum().item()
            train_loss += batch_loss.item()

        # validation
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(tqdm(eval_loader)):
                inputs, labels, index = data['text'], data['intent'], data['id']
                inputs, labels = inputs.to(
                    device), labels.to(device)
                outputs = model(inputs)
                batch_loss = criterion(outputs, labels)
                _, val_pred = torch.max(outputs, 1)
                val_all += labels.shape[0]
                val_acc += (val_pred.cpu() == labels.cpu()).sum().item()
                val_loss += batch_loss.item()

            print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f} | Val Acc: {:3.6f} loss: {:3.6f}'.format(
                epoch + 1, num_epoch, train_acc/train_all, train_loss/len(
                    train_loader), val_acc/val_all, val_loss/len(eval_loader)
            ))

            # 進步則存檔
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), model_path)
                print('saving model with acc {:.3f}'.format(
                    val_acc/val_all))
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
        default="./data/intent/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/intent/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/intent/",
    )

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # optimizer
    parser.add_argument("--lr", type=float, default=3e-4)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda"
    )
    parser.add_argument("--num_epoch", type=int, default=100)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)

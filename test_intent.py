import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import SeqClsDataset
from model import LSTM
from utils import Vocab


def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    data = json.loads(args.test_file.read_text())
    dataset = SeqClsDataset(data, vocab, intent2idx, args.max_len)
    embeddings = torch.load(args.cache_dir / "embeddings.pt")
    max_len = 32
    BATCH_SIZE = 256
    for idx, all_data in enumerate(dataset):  # get data
        result = torch.zeros(max_len, 300)  # template input
        count = 0  # check length
        input_list = all_data['text'].split()
        if len(input_list) > 0:
            for word in input_list:
                if word in vocab.tokens:
                    result[count] = embeddings[vocab.token2idx[word]]
                    count += 1
                    if count >= max_len:
                        break
        all_data['text'] = result

    test_loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False)
    device = get_device()
    print(device)

    input_dim = 300  # glove's dim
    class_num = len(intent2idx)
    hidden = 1024
    layer = 2
    model = LSTM(input_dim, hidden, layer, class_num).to(device)
    model.eval()
    model.load_state_dict(torch.load('./ckpt/intent/best.ckpt'))
    # load weights into model
    result_dict = {}
    model.eval()  # set the model to evaluation mode
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            inputs, index = data['text'], data['id']
            inputs = inputs.to(device)
            outputs = model(inputs)
            # get the index of the class with the highest probability
            _, test_pred = torch.max(outputs, 1)

            for y in range(test_pred.cpu().numpy().shape[0]):
                result_dict[index[y]] = SeqClsDataset.idx2label(
                    dataset, idx=test_pred.cpu().numpy()[y])
    print(len(result_dict))
    with open(args.pred_file, 'w') as f:
        f.write('id,intent\n')
        for i, y in enumerate(result_dict):
            f.write('{},{}\n'.format(y, result_dict[y]))


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--test_file",
        type=Path,
        help="Path to the test file.",
        required=True
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/intent/",
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Path to model checkpoint.",
        required=True
    )
    parser.add_argument("--pred_file", type=Path, default="pred.intent.csv")

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)

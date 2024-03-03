import torch
from torch.optim import Adam
from model.model import WrapperModel, my_collate_fun, MyLoss
from tools.trainer import MultiModalTrainer
from tools.dataset import MultiModalDataset
import argparse
import joblib

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
parser = argparse.ArgumentParser()
parser.add_argument("-m_train_path", type=str, default="/mnt/zgy/dataset/MV")
parser.add_argument("-s_train_path", type=str, default="/mnt/zgy/dataset/Graph")
parser.add_argument("-lr", type=float, default=0.0001)
parser.add_argument("-weight_decay", type=float, default=0.001)
parser.add_argument("-bs", "--batchSize", type=int, help="Batch size for the second stage", default=8)


def main():
    args = parser.parse_args()

    model = WrapperModel('wrapperModel').to(device)
    #model = joblib.load('/mnt/zgy/results/trained_model/multi_modal_17.pkl')
    dataset = MultiModalDataset(args.m_train_path, args.s_train_path)

    train_dataset, val_dataset = dataset.get_train_test_dataset()

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchSize, shuffle=True, num_workers=0,
                                               collate_fn=my_collate_fun, drop_last=True)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batchSize, shuffle=False, num_workers=0,
                                             collate_fn=my_collate_fun, drop_last=True)
    # for param in model.parameters():
    #     print(type(param), param.size())
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    my_loss = torch.nn.CrossEntropyLoss()
    trainer = MultiModalTrainer(model, train_loader, val_loader, optimizer, my_loss)
    trainer.train(50)


if __name__ == '__main__':
    main()

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
from tqdm import tqdm

class ConvPerfPredictor(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, num_layers=4, dropout=0.5,
                    bidirectional=False):
        super().__init__()

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, dropout=dropout,
                            bidirectional=bidirectional)
        D = 2 if bidirectional else 1
        # The linear layer that maps from hidden state space to tag space
        self.hidden2perf1 = nn.Linear(D * hidden_dim, 32)
        self.hidden2perf2 = nn.Linear(32, 16)
        self.hidden2perf3 = nn.Linear(16, 1)

    def forward(self, features):
        lstm_out, _ = self.lstm(features)
        out_last = lstm_out.transpose(0, 1)[-1]
        perf = self.hidden2perf1(out_last)
        perf = self.hidden2perf2(perf)
        perf = self.hidden2perf3(perf)
        return perf


class AnsorScheduleDataset( torch.utils.data.Dataset):

    def __init__(self, schedules, valid_only=False):
        if valid_only:
            self.schedules = [s for s in schedules if s.valid]
        else:
            self.schedules = schedules

        max_len = 0
        max_trans = 0
        for s in schedules:
            max_trans = max(max_trans, len(s.features))
            for f in s.features:
                max_len = max(max_len, len(f))
        self.f_maxlen = max_len
        self.trans_maxlen = max_trans



    def __len__(self):
        return len(self.schedules)

    def __getitem__(self, idx):
        sche = self.schedules[idx]
        features = sche.features
        features_padded = [F.pad(torch.tensor(f, dtype=torch.float),
                                 (0, self.f_maxlen - len(f), )) \
                                            for f in features]
        features_padded = torch.cat([f.unsqueeze(0) for f in features_padded], axis=0)

        return features_padded, torch.tensor(sche.throughput, dtype=torch.float)


def train(schedules, group, outdir, epochs=100, bidirectional=False):

    random.shuffle(schedules)
    split = 0.8
    train_idx = int(0.8 * len(schedules))

    train_dataset = AnsorScheduleDataset(schedules[:train_idx])
    def collate_seqs(batch):
        max_seq = max((len(f) for f in batch))
        new_features = []
        for feature, target in batch:
            new_features.append(\
                    F.pad(feature, (0, 0, 0, max_seq - len(feature)))
                    )
        out = torch.stack(new_features), torch.stack([i[1] for i in batch])
        return out

    train_loader = torch.utils.data.DataLoader(train_dataset,
            batch_size=256, num_workers=8,
            shuffle=True, pin_memory=True,
            collate_fn=collate_seqs)
    val_dataset = AnsorScheduleDataset(schedules[train_idx:])
    val_loader = torch.utils.data.DataLoader(val_dataset,
            batch_size=4096,
            num_workers=8, pin_memory=True,
            collate_fn=collate_seqs)

    hidden_dim = 64
    num_layers = 4
    lstm = ConvPerfPredictor(train_dataset.f_maxlen, hidden_dim, num_layers,
                    bidirectional=bidirectional).cuda()
    print("Training with {} features, train: {}, test: {} on {}"\
            .format(len(schedules), len(train_dataset), len(val_dataset), group))
    print("LSTM with input feature size: {}, hidden: {} X {}"\
                .format(train_dataset.f_maxlen, hidden_dim, num_layers))
    loss_fn = nn.MSELoss()
    optim = torch.optim.AdamW(lstm.parameters())
    loss_curve = []
    accuracy = []
    for epoch in tqdm(range(epochs)):
        #validate
        lstm.eval()
        avg_loss = 0
        count = 0
        correct = 0
        total = 0
        for data, label in val_loader:
            data = data.cuda()
            label = label.cuda()

            pred = lstm(data)
            loss_val = loss_fn(pred, label)
            avg_loss += loss_val.item()
            count += 1

            grid_pred_x, grid_pred_y = torch.meshgrid(pred.flatten(), pred.flatten())
            grid_label_x, grid_label_y = torch.meshgrid(label.flatten(), label.flatten())

            correct_mat = (grid_pred_x >= grid_pred_y) == (grid_label_x >= grid_label_y)

            correct += float(torch.sum(correct_mat))
            total += len(pred) ** 2


        loss_curve.append(avg_loss/count)
        accuracy.append(correct/total)

        if epoch > 0:
            torch.save(lstm, outdir / (group + '_torch_{}.pt'.format(epoch)))
        # train
        lstm.train()
        for data, label in train_loader:
            data = data.cuda()
            label = label.cuda()

            pred = lstm(data)
            loss_val = loss_fn(pred, label)

            loss_val.backward()
            optim.step()
            optim.zero_grad()


    fig = plt.figure()
    ax = fig.gca()
    ax.plot(loss_curve)
    ax.set(ylabel='Validation Loss (MSE)', xlabel="Epochs")
    ax.set(title=group)
    ax.grid(axis='y')
    fig.tight_layout()
    fig.savefig(outdir / (group + '_train.png'))
    plt.close(fig)

    fig = plt.figure()
    ax = fig.gca()
    ax.plot(accuracy)
    ax.set(ylabel='Accuracy', xlabel="Epochs")
    ax.set(title=group)
    ax.grid(axis='y')
    fig.tight_layout()
    fig.savefig(outdir / (group + '_accuracy.png'))
    plt.close(fig)


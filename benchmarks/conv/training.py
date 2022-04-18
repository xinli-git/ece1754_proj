import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
from tqdm import tqdm

class ConvPerfPredictor(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, num_layers=4, dropout=0.5):
        super().__init__()

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, dropout=dropout)
        # The linear layer that maps from hidden state space to tag space
        self.hidden2perf = nn.Linear(hidden_dim, 1)

    def forward(self, features):
        lstm_out, _ = self.lstm(features)
        out_last = lstm_out.transpose(0, 1)[-1]
        perf = self.hidden2perf(out_last)
        return perf


class AnsorScheduleDataset( torch.utils.data.Dataset):

    def __init__(self, schedules):

        self.schedules = schedules
        sample_features = schedules[0].features
        self.f_maxlen = max([len(f) for f in sample_features])

    def __len__(self):
        return len(self.schedules)

    def __getitem__(self, idx):

        features = self.schedules[idx].features
        features_padded = [F.pad(torch.tensor(f, dtype=torch.float),
                                 (0, self.f_maxlen - len(f), )) \
                                            for f in features]
        features_padded = torch.cat([f.unsqueeze(0) for f in features_padded], axis=0)
        return features_padded, torch.tensor( self.schedules[idx].performance, dtype=torch.float)


def train(schedules, group, outdir):

    random.shuffle(schedules)
    split = 0.8
    train_idx = int(0.8 * len(schedules))

    train_dataset = AnsorScheduleDataset(schedules[:train_idx])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, num_workers=8, pin_memory=True)
    val_dataset = AnsorScheduleDataset(schedules[train_idx:])
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4096, num_workers=8, pin_memory=True)
    val_loader2 = torch.utils.data.DataLoader(val_dataset, batch_size=4096, num_workers=8, pin_memory=True)

    hidden_dim = 32
    num_layers = 4
    lstm = ConvPerfPredictor(train_dataset.f_maxlen, hidden_dim, num_layers).cuda()
    print("Training with {} features, train: {}, test: {} on {}"\
            .format(len(schedules), len(train_dataset), len(val_dataset), group))
    print("LSTM with input feature size: {}, hidden: {} X {}"\
                .format(train_dataset.f_maxlen, hidden_dim, num_layers))
    loss_fn = nn.MSELoss()
    optim = torch.optim.AdamW(lstm.parameters())
    loss_curve = []
    accuracy = []
    for epoch in tqdm(range(50)):
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


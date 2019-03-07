from __future__ import print_function
from policy_value_net import PolicyValueNet  # Pytorch
from preprocessing import SelfPlayDataset
from torch.utils import data as dt
import argparse
import torch
from torch.optim import Adam
import torch.nn.functional as F



parser = argparse.ArgumentParser(description='train')
parser.add_argument('--model_path', type=str, default=None, help='Path to model save files.')
parser.add_argument('--model_name', type=str, default=None, help='Model name.')
parser.add_argument('--use_gpu', type=bool, default=True, help='Wheter to use GPU')
parser.add_argument('--selfplay_dir', type=str, default='outputs/train', help='Where to write game data.')
parser.add_argument('--holdout_dir', type=str, default='outputs/valid', help='Where to write held-out game data.')
parser.add_argument('--epochs', type=int, default=1, help='epochs')


def train(args):
    trainset=SelfPlayDataset(args.selfplay_dir, args.model_name)
    validset=SelfPlayDataset(args.holdout_dir, args.model_name)
    trainloader= dt.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=8)
    device = torch.device('cuda:0' if args.use_gpu else 'cpu')
    net=PolicyValueNet(9,9, args.model_path, args.model_name, args.use_gpu).to(device)
    optimizer=Adam(net.parameters())
    print('Training {} on {} examples'.format(args.model_name, len(trainset)))
    for epoch in range(args.epochs):
        for batch_idx, data in enumerate(trainloader):
            features, pi, value = data['features'], data['pi'], data['value']
            features = features.clone().detach().to(device=device , dtype=torch.float)
            pi = pi.clone().detach().to(device=device, dtype=torch.float)
            value = value.clone().detach().to(device=device,dtype=torch.float)
            optimizer.zero_grad()
            pi_out, val_out = net.policy_value_net(features)
            value_loss = F.mse_loss(val_out.view(-1), value)
            policy_loss = -torch.mean(torch.sum(pi * pi_out, 1))
            loss = value_loss + policy_loss
            # backward and optimize
            loss.backward()
            optimizer.step()
        print(epoch, loss)
    net.save_model()
if __name__ == '__main__':
    args = parser.parse_args()
    train(args)

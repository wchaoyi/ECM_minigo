import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import parse
import os

def set_learning_rate(optimizer, lr):
    """Sets the learning rate to the given value"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class BasicBlock(nn.Module):
    expansion =1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride!=1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                                          nn.BatchNorm2d(self.expansion*planes))


    def forward(self, x):
        out=F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Net(nn.Module):
    """policy-value network module"""
    def __init__(self, board_width, board_height, block, num_blocks=[2, 2, 2, 2]):
        super(Net, self).__init__()

        self.board_width = board_width
        self.board_height = board_height
        self.in_planes = 64
        # residual pipeline

        self.conv1 = nn.Conv2d(17, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=1)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=1)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=1)

        self.act_conv1 = nn.Conv2d(512, 4, kernel_size=1)
        self.act_fc1 = nn.Linear(4 * board_width * board_height,
                                 board_width * board_height + 1)
        # state value layers
        self.val_conv1 = nn.Conv2d(512, 2, kernel_size=1)
        self.val_fc1 = nn.Linear(2 * board_width * board_height, 64)
        self.val_fc2 = nn.Linear(64, 1)


        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, state_input):
        # common layers
        out = F.relu(self.bn1(self.conv1(state_input)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        #out = self.avgpool(out)
        # action policy layers
        x_act = F.relu(self.act_conv1(out))
        x_act = x_act.view(-1, 4*self.board_width*self.board_height)
        x_act = F.log_softmax(self.act_fc1(x_act), dim=-1)
        # state value layers
        x_val = F.relu(self.val_conv1(out))
        x_val = x_val.view(-1, 2*self.board_width*self.board_height)
        x_val = F.relu(self.val_fc1(x_val))
        x_val = torch.tanh(self.val_fc2(x_val))
        return x_act, x_val


class PolicyValueNet(torch.nn.Module):
    """policy-value network """
    def __init__(self, board_width, board_height,
                 model_path=None, model_name=None, use_gpu=False):
        super(PolicyValueNet, self).__init__()
        self.model_path = model_path
        self.model_name = model_name
        self.use_gpu = use_gpu
        self.board_width = board_width
        self.board_height = board_height
        self.l2_const = 1e-4  # coef of l2 penalty
        # the policy value net module
        self.policy_value_net = Net(board_width, board_height, BasicBlock) #.cuda()
        format_name='{}_{}'
        self.model_prefix=str(parse.parse(format_name, self.model_name)[0])
        self.model_num=int(parse.parse(format_name, self.model_name)[1])

        if os.path.isfile(os.path.join(model_path, model_name)):
            net_params = torch.load(os.path.join(model_path, model_name), map_location=None if self.use_gpu else 'cpu')
            self.policy_value_net.load_state_dict(net_params)


    def policy_value(self, state_batch):
        """
        input: a batch of states
        output: a batch of action probabilities and state values
        """
        if self.use_gpu:
            state_batch = Variable(torch.FloatTensor(state_batch).cuda())
            log_act_probs, value = self.policy_value_net(state_batch)
            act_probs = np.exp(log_act_probs.data.cpu().numpy())
            return act_probs, value.data.cpu().numpy()
        else:
            state_batch = Variable(torch.FloatTensor(state_batch))
            log_act_probs, value = self.policy_value_net(state_batch)
            act_probs = np.exp(log_act_probs.data.numpy())
            return act_probs, value.data.numpy()

    def policy_value_fn(self, features, device='cuda:0'):
        """
        input: board
        output: a list of (action, probability) tuples for each available
        action and the score of the board state
        """
        if features.ndim == 3:
            features=np.expand_dims(features, 0)
        if self.use_gpu:
            log_act_probs, value = self.policy_value_net(torch.tensor(features).to(device=device, dtype=torch.float))
            act_probs = np.exp(log_act_probs.data.cpu().numpy())
        else:
            log_act_probs, value = self.policy_value_net(
                    Variable(torch.tensor(features.astype('uint8'))).float())
            act_probs = np.exp(log_act_probs.data.numpy())
        return act_probs, value

    def train_step(self, state_batch, mcts_probs, winner_batch, lr):
        """perform a training step"""
        # wrap in Variable
        if self.use_gpu:
            state_batch = Variable(torch.FloatTensor(state_batch).cuda())
            mcts_probs = Variable(torch.FloatTensor(mcts_probs).cuda())
            winner_batch = Variable(torch.FloatTensor(winner_batch).cuda())
        else:
            state_batch = Variable(torch.FloatTensor(state_batch))
            mcts_probs = Variable(torch.FloatTensor(mcts_probs))
            winner_batch = Variable(torch.FloatTensor(winner_batch))

        # zero the parameter gradients
        self.optimizer.zero_grad()
        # set learning rate
        set_learning_rate(self.optimizer, lr)

        # forward
        log_act_probs, value = self.policy_value_net(state_batch)
        # define the loss = (z - v)^2 - pi^T * log(p) + c||theta||^2
        # Note: the L2 penalty is incorporated in optimizer
        value_loss = F.mse_loss(value.view(-1), winner_batch)
        policy_loss = -torch.mean(torch.sum(mcts_probs*log_act_probs, 1))
        loss = value_loss + policy_loss
        # backward and optimize
        loss.backward()
        self.optimizer.step()
        # calc policy entropy, for monitoring only
        entropy = -torch.mean(
                torch.sum(torch.exp(log_act_probs) * log_act_probs, 1)
                )
        return loss.data[0], entropy.data[0]

    def get_policy_param(self):
        net_params = self.policy_value_net.state_dict()
        return net_params

    def save_model(self):
        """ save model params to file """
        net_params = self.get_policy_param()  # get model params
        new_name='{}_{}'.format(self.model_prefix, self.model_num+1)
        torch.save(net_params, os.path.join(self.model_path, new_name))
        print('Saving {}'.format(new_name))



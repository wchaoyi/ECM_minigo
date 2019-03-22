


import argparse
#from policy_value_net import PolicyValueNet
from residual_policy_value_net import PolicyValueNet
import torch
import torch.multiprocessing as mp
from selfplay_utils import run_many_game
import utils



parser = argparse.ArgumentParser(description='selfplay')
parser.add_argument('--model_path', type=str, default=None, help='Path to model save files.')
parser.add_argument('--model_name', type=str, default=None, help='Model name.')
parser.add_argument('--use_gpu', type=bool, default=True, help='Wheter to use GPU')
parser.add_argument('--selfplay_dir', type=str, default='outputs/train', help='Where to write game data.')
parser.add_argument('--holdout_dir', type=str, default='outputs/valid', help='Where to write held-out game data.')
parser.add_argument('--sgf_dir', type=str, default=None, help='Where to write human-readable SGFs.')
parser.add_argument('--holdout_pct', type=float, default= 0.05, help='What percent of games to hold out.')
parser.add_argument('--resign_disable_pct', type=float, default= 0.05,
                    help= 'What percent of games to disable resign for.')
parser.add_argument('--num_processes', type=int, default=1, help= "number of processes to use")
parser.add_argument('--nb_games', type=int, default=1, help= "number of games to play")
parser.add_argument('--verbose', type=int, default=0, help='How much debug info to print.')





if __name__ == '__main__':
    args = parser.parse_args()
    mp.set_start_method('spawn')
    device1 = torch.device('cuda:0' if args.use_gpu else 'cpu')
    device2 = torch.device('cuda:1' if args.use_gpu else 'cpu')
    with utils.logged_timer("Loading weights from %s ... " % args.model_name):
        network1 = PolicyValueNet(9, 9, args.model_path, args.model_name, args.use_gpu).to(device1)
        network2 = PolicyValueNet(9, 9, args.model_path, args.model_name, args.use_gpu).to(device2)
    processes = []

    for rank in range(args.num_processes):
        p = mp.Process(target=run_many_game, args=(network1 if rank<args.num_processes/2 else network2,
                                                  args, device1 if rank<args.num_processes/2 else device2))
        #p = mp.Process(target=run_many_game, args=(network2,
         #                                          args, device2))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()


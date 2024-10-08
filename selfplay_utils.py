import random
import os
import time
import numpy as np
import coords
import strategies
from strategies import MCTSPlayer
import utils
from features import extract_features, NEW_FEATURES
import preprocessing

strat_args = strategies.main('')


def play(network, args, device=None):
    ''' Plays out a self-play match, returning a MCTSPlayer object containing:
        - the final position
        - the n x 362 tensor of floats representing the mcts search probabilities
        - the n-ary tensor of floats representing the original value-net estimate
          where n is the number of moves in the game'''
    readouts = strat_args.num_readouts  # defined in strategies.py
    # Disable resign in 5% of games
    if random.random() < args.resign_disable_pct:
        resign_threshold = -1.0
    else:
        resign_threshold = None

    player = MCTSPlayer(network, device=device, resign_threshold=resign_threshold)

    player.initialize_game()

    # Must run this once at the start to expand the root node.
    first_node = player.root.select_leaf()
    features = extract_features(first_node.position, NEW_FEATURES)
    prob, val = network.policy_value_fn(features, device=device)
    first_node.incorporate_results(prob.flatten(), val.flatten(), first_node)

    while True:
        start = time.time()
        player.root.inject_noise()
        current_readouts = player.root.N
        # we want to do "X additional readouts", rather than "up to X readouts".
        while player.root.N < current_readouts + readouts:
            player.tree_search()

        if args.verbose >= 3:
            print(player.root.position)
            print(player.root.describe())

        if player.should_resign():
            player.set_result(-1 * player.root.position.to_play,
                              was_resign=True)
            break
        move = player.pick_move()
        player.play_move(move)
        if player.root.is_done():
            player.set_result(player.root.position.result(), was_resign=False)
            break

        if (args.verbose >= 2) or (args.verbose >= 1 and player.root.position.n % 10 == 9):
            print("Q: {:.5f}".format(player.root.Q))
            dur = time.time() - start
            print("%d: %d readouts, %.3f s/100. (%.2f sec)" % (
                player.root.position.n, readouts, dur / readouts * 100.0, dur), flush=True)
        if args.verbose >= 3:
            print("Played >>",
                  coords.to_gtp(coords.from_flat(player.root.fmove)))

    if args.verbose >= 2:
        utils.dbg("%s: %.3f" % (player.result_string, player.root.Q))
        utils.dbg(player.root.position, player.root.position.score())

    return player


def run_game(network, args, device=None, sgf_dir=None, holdout_pct=0.05):
    '''Takes a played game and record results and game data.'''
    selfplay_dir = os.path.join(args.selfplay_dir, args.model_name)
    utils.ensure_dir_exists(selfplay_dir)
    holdout_dir = os.path.join(args.holdout_dir, args.model_name)
    utils.ensure_dir_exists(holdout_dir)
    if args.sgf_dir:
        sgf_dir = os.path.join(args.sgf_dir, args.model_name)
        utils.ensure_dir_exists(sgf_dir)
    if sgf_dir is not None:
        minimal_sgf_dir = os.path.join(sgf_dir, 'clean')
        full_sgf_dir = os.path.join(sgf_dir, 'full')
        utils.ensure_dir_exists(minimal_sgf_dir)
        utils.ensure_dir_exists(full_sgf_dir)
    if selfplay_dir is not None:
        utils.ensure_dir_exists(selfplay_dir)
        utils.ensure_dir_exists(holdout_dir)

    with utils.logged_timer("Playing game"):
        player = play(network, args, device=device)

    features, pis, values = player.extract_data(return_features=True)
    features = np.array(features)
    pis = np.array(pis)
    values = np.array(values)
    assert features.shape[0] == pis.shape[0] == values.shape[0]
    output_name = '{}-{}'.format(int(time.time()), features.shape[0])
    if sgf_dir is not None:
        with open(os.path.join(minimal_sgf_dir, '{}.sgf'.format(output_name)), 'w') as f:
            f.write(player.to_sgf(use_comments=False))
        with open(os.path.join(full_sgf_dir, '{}.sgf'.format(output_name)), 'w') as f:
            f.write(player.to_sgf())

    if selfplay_dir is not None:
        # Hold out 5% of games for validation.
        if random.random() < holdout_pct:
            fname = os.path.join(holdout_dir,
                                 "{}.hdf5".format(output_name))
        else:
            fname = os.path.join(selfplay_dir,
                                 "{}.hdf5".format(output_name))

        preprocessing.save_h5_examples(fname, features, pis, values)


def run_many_game(network, args, device=None):
    for game in range(int(args.nb_games / args.num_processes)):
        run_game(network, args, device=device, holdout_pct=args.holdout_pct)

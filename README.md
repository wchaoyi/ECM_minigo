ECM_Minigo: A minimalist Go engine modeled after AlphaGo Zero in Pytorch, built on Tensorflow/minigo
==================================================
This is a student-led PyTorch re-implementation of minigo project : 
https://github.com/tensorflow/minigo. Here is the description of their project : \
This is an implementation of a neural-network based Go AI, using PyTorch.


Repeat, *this is not the official AlphaGo program by DeepMind*.  This is an


ECM_Minigo is based off Tensorflow's "[MiniGo](https://github.com/tensorflow/minigo/)"
-- a pure Python implementation of AlphaGo Zero paper ["Mastering the Game of Go without Human
Knowledge"](https://www.nature.com/articles/nature24270). More recently, this
architecture was extended for Chess and Shogi in ["Mastering Chess and Shogi by
Self-Play with a General Reinforcement Learning
Algorithm"](https://arxiv.org/abs/1712.01815).  These papers will often be
abridged in this documentation *AGZ* (for AlphaGo
Zero), and *AZ* (for AlphaZero) respectively.


Goals of the Project
==================================================

1. Understand clearly how AGZ works, and be able in our future professional life 
to transpose what we learned to pur work

2. Reproduce the methods of the original DeepMind AlphaGo Zero paper as faithfully
   as possible, through an open-source implementation and open-source pipeline
   tools.

An explicit non-goal of the project is to produce a competitive Go program that
establishes itself as the top Go AI. Instead, we strive for a readable,
understandable implementation that can benefit the community, even if that
means our implementation is not as fast or efficient as possible.

While this product might produce such a strong model, we hope to focus on the
process.  Remember, getting there is half the fun. :)

We hope this project is an accessible way for interested developers to have
access to a strong Go model with an easy-to-understand platform of python code
available for extension, adaptation, etc.




Selfplay
--------
To watch Minigo play a game, you need to specify a model. Here's an example
to play using the latest model in your bucket

```shell
python3 selfplay.py --use_gpu=True --num_processes=4 \
--nb_games=10000 --model_path=models \
--model_name=resnet18_9 --sgf_dir=outputs/sgf\
```

where `num_processes` is for optimising the time by playing many games 
at the same time

Playing Against Minigo
----------------------

Minigo uses the
[GTP Protocol](http://www.lysator.liu.se/~gunnar/gtp/gtp2-spec-draft2/gtp2-spec.html),
and you can use any gtp-compliant program with it.

```shell
python3 gtp.py --model_path=models \
--model_name=resnet18_10 --use_gpu=True'
```

After some loading messages, it will display `GTP engine ready`, at which point
it can receive commands.  GTP cheatsheet:

```
genmove [color]             # Asks the engine to generate a move for a side
play [color] [coordinate]   # Tells the engine that a move should be played for `color` at `coordinate`
showboard                   # Asks the engine to print the board.
```

One way to play via GTP is to use gogui-display (which implements a UI that
speaks GTP.) You can download the gogui set of tools at
[http://gogui.sourceforge.net/](http://gogui.sourceforge.net/). See also
[documentation on interesting ways to use
GTP](http://gogui.sourceforge.net/doc/reference-twogtp.html).

```shell
gogui-twogtp -black 'python3 gtp.py --model_path=models \
--model_name=resnet18_10 --use_gpu=True'' -white 'gogui-display' -size 9 -komi 7.5 -verbose -auto
```

Another way to play via GTP is to watch it play against GnuGo, while
spectating the games:

```shell
BLACK="gnugo --mode gtp"
WHITE="python3 gtp.py --model_path=models \
--model_name=resnet18_10 --use_gpu=True'"
TWOGTP="gogui-twogtp -black \"$BLACK\" -white \"$WHITE\" -games 10 \
  -size 19 -alternate -sgffile gnugo"
gogui -size 19 -program "$TWOGTP" -computer-both -auto
```

Training Minigo
======================

Overview
--------

The following sequence of commands will allow you to do one iteration of
reinforcement learning on 9x9. These are the basic commands used to produce the
models and games referenced above.


 - selfplay: plays games with the latest model, producing data used for training. Game
 data is saved as hdf5 files.
 - train: trains a new model with the selfplay results from the most recent N
   generations. It uses PyTorch DataSet and DataLoader to efficiently read and process 
   the data.


```

Self-play
---------

This command starts self-playing, outputting its raw game data as tf.Examples
as well as in SGF form in the directories.


```shell
python3 selfplay.py --use_gpu=True --num_processes=4 --nb_games=10000 \
--model_path=models --model_name=resnet18_9 --sgf_dir=outputs/sgf
```

Training
--------

This command takes the model name of the model you want to train, assuming selfplay data has already 
been gererated and run and trains and saves a
new model.

Run the training job:

```shell
python3 train_from_disk.py --model_path=models --model_name=resnet18_10 --epochs=40

```







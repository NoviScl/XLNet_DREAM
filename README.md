## XLNet baseline for DREAM dataset 
Author: Chenglei Si (River Valley High School, Singapore)

Usage:

1. Download data and unzip to this folder.
2. (If you have not installed sentencepiece) Run `pip install sentencepiece`
3. Run `sh run.sh`

(The hyperparameters that I used can be found in run.sh)

Result: 72.0 (SOTA as of July 2019, [leaderboard](https://dataset.org/dream/))

Note: My codes are built upon huggingface's implementation of [pytorch_transformers](https://github.com/huggingface/pytorch-transformers), and the original XLNet paper is: [(Yang et al., 2019)](https://arxiv.org/pdf/1906.08237.pdf).



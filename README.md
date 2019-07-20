Usage:

1. Download data and unzip to this folder.
2. pip install sentencepiece 
3. takes a while to download the pretrained XLNet model
4. Use apex and fp16 to speed up training. 
5. Single GPU may not be able to to fit a batch, recommend using multi GPU 
6. may need to fix apex bug: https://github.com/NVIDIA/apex/issues/131

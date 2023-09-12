# KEST
Official script of IJCAI 2023 paper: [KEST: Kernel Distance Based Efficient Self-Training for Improving Controllable Text Generation](https://www.ijcai.org/proceedings/2023/561).
Full version of our paper can be found on [arxiv](https://arxiv.org/abs/2306.10414).

## Introduction
Self-training (ST) has come to fruition in language understanding tasks by producing pseudo labels, which reduces the labeling bottleneck of language model fine-tuning. Nevertheless, in facilitating semi-supervised controllable language generation, ST faces two key challenges. First, augmented by self-generated pseudo text, generation models tend to over-exploit the previously learned text distribution, suffering from mode collapse and poor generation diversity. Second, generating pseudo text in each iteration is time-consuming, severely decelerating the training process. In this work, we propose KEST, a novel and efficient self-training framework to handle these problems. KEST utilizes a kernel-based loss, rather than standard cross entropy, to learn from the soft pseudo text produced by a shared non-autoregressive generator. We demonstrate both theoretically and empirically that KEST can benefit from more diverse pseudo text in an efficient manner, which allows not only refining and exploiting the previously fitted distribution but also enhanced exploration towards a larger potential text space, providing a guarantee of improved performance. Experiments on three controllable generation tasks demonstrate that KEST significantly improves control accuracy while maintaining comparable text fluency and generation diversity against several strong baselines.


## Repository
```
KEST
├── data
├── corpus
├── codes
├── (unilm)
└── (your evaluation classifier)
```

## Data
You can download the training data of [IMDb](https://huggingface.co/datasets/imdb), [AGNews](https://huggingface.co/datasets/ag_news) from Huggingface. [Jigsaw](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/) dataet can be found on Kaggle.

We use UniLM1-base-cased for our base model. Please download it from the following [link](https://github.com/microsoft/unilm/tree/master/unilm-v1).

## Training
This code can be ran with single GPU. Script that works on multi GPU is on process. 

Simply run [train.py](codes/train.py) to replicate our experimental result.

You are free to play with the hyperparameters and settings in [config.py](codes/config.py).

## Evaluation/Inference
[evaluation.py](codes/evaluation.py) evaluates the classification performance of trained model (F1) and generalizability of generation (Model PPL).

[generation.py](codes/generation.py) generates samples of given prompt and evaluates the fluency (Output PPL), classification, and diversity (Dist, Self-BLEU).


## License

This repository is licensed under the [MIT License](LICENSE). 

## Citation

If you find our work useful, please consider citing our IJCAI paper:

```
@inproceedings{feng-et-al-2023-kest,
  title     = {KEST: Kernel Distance Based Efficient Self-Training for Improving Controllable Text Generation},
  author    = {Feng, Yuxi and Yi, Xiaoyuan and Lakshmanan, Laks V.S. and Xie, Xing},
  booktitle = {Proceedings of the Thirty-Second International Joint Conference on
               Artificial Intelligence, {IJCAI-23}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  editor    = {Edith Elkind},
  pages     = {5049--5057},
  year      = {2023},
  month     = {8},
  note      = {Main Track},
  doi       = {10.24963/ijcai.2023/561},
  url       = {https://doi.org/10.24963/ijcai.2023/561},
}
```

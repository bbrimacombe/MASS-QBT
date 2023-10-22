# QBT: Quick Back-Translation for Unsupervised Machine Translation

![alt text](https://github.com/bbrimacombe/Quick-Back-Translation/blob/master/figs/qbt.jpg)

# Overview
Quick back-translation makes a simple change to the Tranformer architecture during unsupervised machine translation training. The encoder, a non-autoregressive model, is repurposed as a geneative model to supplement back-translation. This enhancement requires no changes to the model architecture, and any encoder-decoder Transformer can be trained with QBT.

For the QBT method we have updated:
- /train.py
- /src/trainer.py
- /src/model/transformer.py
- /src/data/loader.py
- /src/evaluation/evaluator.py  

For local:
python train.py --exp_name unsupMT_enfr --data_path ./data/processed/en-fr/ --lgs en-fr --encoder_only false --emb_dim 1024 --n_layers 6 --n_heads 8 --dropout 0.1 --attention_dropout 0.1 --gelu_activation true --tokens_per_batch 2000 --batch_size 32 --bptt 256 --optimizer adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.0001 --epoch_size 10 --max_epoch 30000 --eval_bleu true --reload_model mass_ft_enfr_1024.pth,mass_ft_enfr_1024.pth --qbt_steps en-fr-en,fr-en-fr --encoder_bt_steps en-fr-en,fr-en-fr ...

For distributed:
export NGPU=8; CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=$NGPU train.py --exp_name unsupMT_ende --data_path ./data/processed/de-en/ --lgs en-de --encoder_only false --emb_dim 1024 --n_layers 6 --n_heads 8 --dropout 0.1 --attention_dropout 0.1 --gelu_activation true --tokens_per_batch 1000 --batch_size 16 --bptt 256 --optimizer adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.0001 --epoch_size 20000 --max_epoch 30000 --eval_bleu true --reload_model mass_ft_ende_1024.pth,mass_ft_ende_1024.pth --bt_steps en-de-en,de-en-de --qbt_steps en-de-en,de-en-de --encoder_bt_steps en-de-en,de-en-de --encoder_mt_random_steps en-de,de-en ...

Note: Inside trainer.py, we can toggle the gradient during BT -- if we would like to update the encoder weights or not. For QBT-Staged, typically we *do not* update encoder gradients until BT.


### Dependencies
Currently QBT is implemented for unsupervised NMT based on the codebase of [XLM](https://github.com/facebookresearch/XLM), and subsequently [MASS](https://github.com/facebookresearch/XLM](https://github.com/ishwnews/MASS). The depencies are as follows:
- Python 3
- NumPy
- PyTorch (version 0.4 and 1.0)
- fastBPE (for BPE codes)
- Moses (for tokenization)
- Apex (for fp16 training)

### Data

We use the same BPE codes and vocabulary as XLM and MASS. To download the datasets the commands can be found in the XLM or MASS codebases. For example in English and French:

```
cd MASS-QBT

wget https://dl.fbaipublicfiles.com/XLM/codes_enfr
wget https://dl.fbaipublicfiles.com/XLM/vocab_enfr

./get-data-nmt.sh --src en --tgt fr --reload_codes codes_enfr --reload_vocab vocab_enfr
```

### Pre-training:

#### Distributed Training

To use *multiple GPUs* e.g. 3 GPUs **on same node**
```
export NGPU=3; CUDA_VISIBLE_DEVICES=0,1,2 python -m torch.distributed.launch --nproc_per_node=$NGPU train.py [...args]
```
To use *multiple GPUS* across **many nodes**, use Slurm to request multi-node job and launch the above command. 
The code automatically detects the SLURM_* environment vars to distribute the training.


### Fine-tuning 
After pre-training, we use back-translation to fine-tune the pre-trained model on unsupervised machine translation:
```
MODEL=mass_enfr_1024.pth

python train.py \
  --exp_name unsupMT_enfr                              \
  --data_path ./data/processed/en-fr/                  \
  --lgs 'en-fr'                                        \
  --bt_steps 'en-fr-en,fr-en-fr'                       \
  --encoder_only false                                 \
  --emb_dim 1024                                       \
  --n_layers 6                                         \
  --n_heads 8                                          \
  --dropout 0.1                                        \
  --attention_dropout 0.1                              \
  --gelu_activation true                               \
  --tokens_per_batch 2000                              \
  --batch_size 32	                                     \
  --bptt 256                                           \
  --optimizer adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.0001 \
  --epoch_size 200000                                  \
  --max_epoch 30                                       \
  --eval_bleu true                                     \
  --reload_model "$MODEL,$MODEL"                       \
```


| Model | Ro-En BLEU (with BT) |
|:---------:|:----:|
| Baseline | 34.0 |
| XLM | 38.5 |
| [MASS](https://modelrelease.blob.core.windows.net/mass/mass_mt_enro_1024.pth) | 39.1 |



## Reference

If you find QBT useful in your research, you may cite the paper as:

...
    



# Smoothed-ModernBERT (ANONYMOUS PyTorch implementation)
This is an ANONYMOUS PyTorch implementation of Smoothed-ModernBERT: Co-Attentional Synergy of Probabilistic Topic Models and ModernBERT through Dynamic Fusion

## PLEASE NOTE THAT THIS REPO WILL BE MOVED TO OUR OFFICIAL GitHub after the paper acceptance (paper is currently under review)


### Getting Started:

To install conda packge: [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html)

Then install the requirements by running:
```
conda env create -f environment.yml
```

This will create a Python environment and install all the necessary packages required to run the scripts without distrupting your machine packages.

To train and test the model, run `python main.py` 


Sample `config.json`:

```js
{
    "dataset": "reuters8",
    "label_path": ".../labels.txt",
    "train_dataset_path": ".../training.tsv",
    "val_dataset_path": ".../validation.tsv",
    "test_dataset_path": ".../test.tsv",
    "num_workers": 8,
    "batch_size": 16,
    "warmup_steps": 10,
    "lr": 2e-05,
    "alpha": 0.9,
    "num_epochs": 20,
    "clip": 1.0,
    "seed": 42,
    "device": "cuda",
    "val_freq": 0.0,
    "test_freq": 0.0,
    "disable_tensorboard": false,
    "tensorboard_dir": "runs/topicbert-512",
    // directory where checkpoints should be
    "resume": ".../checkpoints/", 
    // whether to look for a checkpoint in above or just save a new one there
    "save_checkpoint_only": true, 
    "verbose": true,
    "silent": false,
    "load": null,
    "save": "config.json"
}
```


## Data sets
We use five data sets and we have set Reuter r8 as a defaul. To use other data sets, please download it [DATA SETs](https://disi.unitn.it/moschitti/corpora.htm) in the same directory named 'raw_data'  



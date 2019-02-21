# Pytorch implementation of [A simple neural network module for relational reasoning](https://arxiv.org/pdf/1706.01427.pdf)

## Requirements

- Python3
- Pytorch 1.0.0
- TensorBoardX

## Usage

generate sort-of-clevr dataset
```
python soc_generator.py
```

train
```
python train.py 
    --batch_size [64]
    --n_epoch [120]
    --lr [1e-4]
    --weight_decay [1e-4]
    --save_dir [model]
    --dataset [data/sort-of-clevr.pickle]
    --init [kaiming]
    --resume []
    --model_type [light_light_light]
    --seed [12345]
    --n_cpu [4]
```

test
```
python test.py
    --model_type
    --dataset
    --model
```

## Result

| Sort-of-CLEVR | light_light_light | heavy_heavy_heavy | patch_light_light |
|---------------|-------------------|-------------------|-------------------|
| Accuracy      | 94%               | 94%               | 97%               |

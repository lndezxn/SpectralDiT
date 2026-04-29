# SpectralDiT

```bash
python train.py --config configs/cifar10_dit_small.yaml
```

```bash
accelerate launch train.py --config configs/cifar10_dit_small.yaml
```

```bash
python train.py --config configs/cifar10_dit_small.yaml
# set train.resume_from to outputs/cifar10_dit_small/<run_timestamp>/checkpoints/step_0010000
```

```bash
python sample.py --config configs/cifar10_dit_small.yaml --ckpt outputs/cifar10_dit_small/checkpoints/step_0001000/checkpoint.pt
```

```bash
tensorboard --logdir outputs/cifar10_dit_small
```

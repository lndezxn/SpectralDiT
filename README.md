# SpectralDiT

```bash
python train.py --config configs/cifar10_dit_small.yaml
```

```bash
accelerate launch train.py --config configs/cifar10_dit_small.yaml
```

```bash
tensorboard --logdir outputs/cifar10_dit_small/tensorboard
```

```bash
python sample.py --config configs/cifar10_dit_small.yaml --ckpt outputs/cifar10_dit_small/checkpoints/step_0001000/checkpoint.pt
```

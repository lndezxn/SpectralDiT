# SpectralDiT

```bash
python scripts/train.py --config configs/cifar10_dit_small.yaml
```
Start single-process training with the default DiT-small config.

```bash
accelerate launch scripts/train.py --config configs/cifar10_dit_small.yaml
```
Start training through `accelerate`, for multi-GPU or managed mixed precision runs.

```bash
python scripts/train.py --config configs/cifar10_dit_small.yaml
# set train.resume_from to outputs/cifar10_dit_small/<run_timestamp>/checkpoints/step_0010000
```
Resume training from an existing checkpoint by setting `train.resume_from` in the config first.

```bash
python scripts/sample.py --config configs/cifar10_dit_small.yaml --ckpt outputs/cifar10_dit_small/checkpoints/step_0001000/checkpoint.pt
```
Sample images from a checkpoint and save the image grid under `train.output_dir/manual_samples`.

```bash
python scripts/sample.py --config configs/cifar10_dit_small.yaml --ckpt outputs/cifar10_dit_small/checkpoints/step_0001000/checkpoint.pt --label 3
```
Sample only a specified class label instead of cycling through all classes.

```bash
python scripts/sample.py --config configs/cifar10_dit_small.yaml --ckpt outputs/cifar10_dit_small/checkpoints/step_0001000/checkpoint.pt
# set sample.debug.enabled=true to dump per-step token debug tensors under manual_samples/debug_tokens
```
Sample with debug dumping enabled to save per-step token and pixel-space intermediates.

```bash
python scripts/visualize_debug.py --input outputs/cifar10_dit_small/manual_samples/debug_tokens
```
Render PNG visualizations from saved debug `.pt` dumps.

```bash
tensorboard --logdir outputs/cifar10_dit_small
```
Open TensorBoard for training curves and evaluation previews.

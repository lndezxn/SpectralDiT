# SpectralDiT

## Training

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

## Sampling

```bash
python scripts/sample.py --config configs/cifar10_dit_small.yaml --ckpt outputs/cifar10_dit_small/checkpoints/step_0001000/checkpoint.pt
```
Sample images from a checkpoint and save the image grid under `train.output_dir/manual_samples`.

```bash
python scripts/sample.py --config configs/cifar10_dit_small.yaml --ckpt outputs/cifar10_dit_small/checkpoints/step_0001000/checkpoint.pt --label 3
```
Sample only a specified class label instead of cycling through all classes.

```bash
python scripts/visualize_checkpoints.py \
  --ckpt outputs/run_a/checkpoints/step_0100000/checkpoint.pt outputs/run_b/checkpoints/step_0100000/checkpoint.pt \
  --row-labels run_a run_b \
  --num-samples 8 \
  --class-label 3
```
Sample multiple checkpoints with each checkpoint run's `config_resolved.yaml`, then save a compact row-per-checkpoint comparison grid with vertical labels under the first config's `train.output_dir/manual_samples`.

## Evaluation

```bash
python scripts/evaluate_dit.py --ckpt outputs/cifar10_dit_small/checkpoints/step_0001000/checkpoint.pt
```
Evaluate a checkpoint with FID, Inception Score, precision/recall, Fourier spectrum, high-frequency energy, and edge statistics using 10000 generated and 10000 real samples by default. The script loads `config_resolved.yaml` from the checkpoint case directory unless `--config` is provided.

```bash
python scripts/plot_radial_spectrum.py outputs/metrics/patch_1.json outputs/metrics/patch_4.json --output outputs/metrics/radial_spectrum.png
```
Plot real and generated radial Fourier spectra from one or more evaluation JSON files, using normalized radial frequency on the x-axis.

## Debug Visualization

```bash
python scripts/sample.py --config configs/cifar10_dit_small.yaml --ckpt outputs/cifar10_dit_small/checkpoints/step_0001000/checkpoint.pt
# set sample.debug.enabled=true to dump per-step token debug tensors under manual_samples/debug_tokens
```
Sample with debug dumping enabled to save per-step token and pixel-space intermediates.

```bash
python scripts/visualize_debug.py --input outputs/cifar10_dit_small/manual_samples/debug_tokens
```
Render PNG visualizations from saved debug `.pt` dumps.
Per-sample raw feature-map grids, per-block feature-map PNGs, and pixel-space PNGs are written under each sample's `raw_images/` directory.

```bash
python scripts/visualize_gate_evolution.py --run-dir outputs/cifar10_spectraldit_small/<run_timestamp>
```
Collect frequency gate activations across all checkpoints in a run and render per-block low/high gate heat maps.

```bash
python scripts/visualize_gate_evolution.py --run-dir outputs/cifar10_spectraldit_small/<run_timestamp> --plot-only
```
Redraw gate heat maps from the saved cache without rerunning checkpoint inference.

```bash
python scripts/visualize_gate_evolution.py --run-dir outputs/cifar10_spectraldit_small/<run_timestamp> --plot-only --step-min 100000 --step-max 300000 --max-x-ticks 6 --max-y-ticks 8
```
Redraw gate heat maps for a training-step range with sparser training-step and sampling-time labels.

## Monitoring

```bash
tensorboard --logdir outputs/cifar10_dit_small
```
Open TensorBoard for training curves and evaluation previews.

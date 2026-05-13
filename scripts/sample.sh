python scripts/visualize_checkpoints.py \
  --ckpt \
    outputs/cifar10_dit_small/patch_1/checkpoints/step_0400000/checkpoint.pt \
    outputs/cifar10_spectraldit_small/patch_1_time_1.0/checkpoints/step_0400000/checkpoint.pt \
  --row-labels DiT SpectralDiT \
  --num-samples 16 \
  --output outputs/metrics/sample.png
python scripts/evaluate_dit.py \
    --ckpt outputs/cifar10_dit_small/patch_1/checkpoints/step_0400000/checkpoint.pt \
    --output outputs/metrics/patch_1.json

python scripts/evaluate_dit.py \
    --ckpt outputs/cifar10_spectraldit_small/patch_1_time_1.0/checkpoints/step_0400000/checkpoint.pt \
    --output outputs/metrics/patch_1_time_1.0.json
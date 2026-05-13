# python scripts/evaluate_dit.py \
#     --ckpt outputs/cifar10_dit_small/patch_2/checkpoints/step_0400000/checkpoint.pt \
#     --hf-threshold 0.5 \
#     --output outputs/metrics/patch_2.json

# python scripts/evaluate_dit.py \
#     --ckpt outputs/cifar10_spectraldit_small/patch_2_time_1.0/checkpoints/step_0400000/checkpoint.pt \
#     --hf-threshold 0.5 \
#     --output outputs/metrics/patch_2_time_1.0.json

# python scripts/evaluate_dit.py \
#     --ckpt outputs/cifar10_dit_small/patch_4/checkpoints/step_0400000/checkpoint.pt \
#     --hf-threshold 0.5 \
#     --output outputs/metrics/patch_4.json

python scripts/evaluate_dit.py \
    --ckpt outputs/cifar10_spectraldit_small/patch_1_time+label_1.0/checkpoints/step_0400000/checkpoint.pt \
    --hf-threshold 0.5 \
    --output outputs/metrics/patch_1_time+label_1.0.json
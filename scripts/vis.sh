rm -rf outputs/cifar10_spectraldit_small_sample/manual_samples/

python scripts/sample.py --config configs/cifar10_spectraldit_small_sample.yaml \
    --ckpt outputs/cifar10_spectraldit_small/20260505_064639/checkpoints/step_0220000/checkpoint.pt \

python scripts/visualize_debug.py --input outputs/cifar10_spectraldit_small_sample/manual_samples/debug_tokens
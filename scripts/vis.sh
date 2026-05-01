rm -rf outputs/cifar10_dit_small_sample/manual_samples/

python scripts/sample.py --config configs/cifar10_dit_small_sample.yaml \
    --ckpt outputs/cifar10_dit_small/20260430_045544/checkpoints/step_0360000/checkpoint.pt \
    --label 9

python scripts/visualize_debug.py --input outputs/cifar10_dit_small_sample/manual_samples/debug_tokens
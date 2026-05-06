# Test Code

Test scripts for validating the public TexasPoker policy stack.

## Quick Start

```bash
conda activate texas
python test_code/check_environment.py
python test_code/test_training.py
python test_code/test_rdt.py
```

## Main Tests

`test_training.py` generates a tiny synthetic dataset, runs a few epochs through `train.py`, and verifies the training path does not crash. It covers the public model families:

| Scenario | Model |
| --- | --- |
| `dp_pos_only` | Diffusion Policy UNet |
| `dp_resnet_img` | Diffusion Policy with RGB |
| `dp_multitask` | Diffusion Policy with instruction ids |
| `dp_dinov2_transformer` | Diffusion Policy Transformer with DinoV2 features |
| `act_pos_only` | ACT |
| `baku_pos_only` | BAKU |
| `baku_resnet_multitask` | BAKU with RGB and instruction ids |
| `rdt_pos_only` | RDT |
| `rdt_pos_multitask` | RDT with instruction ids |
| `rdt_precomputed_multitask` | RDT with SigLIP patch features |

List scenarios with:

```bash
python test_code/test_training.py --list
```

`test_rdt.py` contains RDT unit and integration checks. `test_deploy.py` and `test_deploy_real_data.py` smoke-test the ZeroMQ deployment server.

Utility scripts such as `print_npz_content.py`, `visualize_npz_image.py`, and `verify_obs_consistency.py` are kept for dataset inspection.

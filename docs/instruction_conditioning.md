# Instruction Conditioning

TexasPokerRobot contains 14 manipulation tasks. In multi-task training, each sample carries an integer `instruction` id extracted from the task directory, and `workflow/instructions.json` maps that id to a human-readable name and text prompt.

## Modes

`train.py` supports two mutually exclusive instruction modes:

- `--use_instruction`: integer task ids are embedded with `nn.Embedding`.
- `--use_text_instruction`: task text is encoded by a frozen text encoder and projected into the policy condition space.

The public training scripts use integer ids by default for DP, ACT, BAKU, and RDT reproduction. RDT also has its own T5 language pathway, configured by the `--rdt_text_encoder`, `--rdt_token_max_len`, and `--instructions_file` flags.

## Data Flow

1. The data loader returns `batch["instruction"]` with shape `(B,)`.
2. `ObsEncoder` adds instruction features when requested.
3. The selected policy consumes those features through its normal observation path.
4. Checkpoints store the observation encoder config, model config, and normalization stats needed for deployment.

## Model Notes

| Model | Registry name | Instruction path |
| --- | --- | --- |
| Diffusion Policy | `diffusion_policy` | `ObsEncoder` feature conditioning; optional dedicated denoiser token |
| ACT | `act` | instruction token in the transformer token stream |
| BAKU | `baku` | instruction features plus optional FiLM for compatible ResNet branches |
| RDT | `rdt` | RDT language/image/action condition streams with task text lookup |

## Common Commands

```bash
bash scripts/train_dp.sh data/easy_mode checkpoints/dp 0 data/vitl14_features
bash scripts/train_act.sh data/easy_mode checkpoints/act 0
bash scripts/train_baku.sh data/easy_mode checkpoints/baku 0
bash scripts/train_rdt.sh data/easy_mode checkpoints/rdt 0 data/siglip_features
```

For custom task text, edit `workflow/instructions.json` while preserving the numeric keys used by the dataset layout.

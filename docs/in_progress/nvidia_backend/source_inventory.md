# Restart Source Corpus Inventory

Acquisition timestamp: 2026-06-20T03:54:30Z.

This inventory records the external source corpus requested by `RESTART.md`.
Sources were acquired into repo-relative ignored paths under
`tmp/restart-sources/` from a clean `source-inventory` worktree based on
`origin/main`.

## Command Patterns

- GitHub organization enumeration:
  `gh api --paginate /orgs/hw-native-sys/repos`
- Repository clone pattern:
  `git clone --depth 1 --filter=blob:none <url> <destination>`
- Page and paper download pattern:
  `curl -fL --retry 3 --retry-delay 2 -o <destination> <url>`

All commands below completed with exit status 0 unless otherwise noted.

## `hw-native-sys` Repositories

The current GitHub org enumeration returned these repositories:

| Source | Destination | HEAD SHA |
| --- | --- | --- |
| `hw-native-sys/pypto` | `tmp/restart-sources/hw-native-sys/pypto` | `2cd14c2cb076730a72233aeb14a2bc623ff9623f` |
| `hw-native-sys/.github` | `tmp/restart-sources/hw-native-sys/.github` | `66d2246804ca885631aff85883d4398c81eec159` |
| `hw-native-sys/simpler` | `tmp/restart-sources/hw-native-sys/simpler` | `24f92a8ac45e3ed21fa8d1b59adce80b806abf75` |
| `hw-native-sys/PTOAS` | `tmp/restart-sources/hw-native-sys/PTOAS` | `da011a3d178ebc882741c88935a90f7ccb159520` |
| `hw-native-sys/pto-isa` | `tmp/restart-sources/hw-native-sys/pto-isa` | `e25732f09943dc53e6d85d68bdcfe653b1922f69` |
| `hw-native-sys/pypto-lib` | `tmp/restart-sources/hw-native-sys/pypto-lib` | `fc2a3da003798f957e66e1e85d5e80b1b718784d` |
| `hw-native-sys/pypto_top_level_documents` | `tmp/restart-sources/hw-native-sys/pypto_top_level_documents` | `69fe802cc263e4bee27241568da9826540a2fb8e` |
| `hw-native-sys/pypto-serving` | `tmp/restart-sources/hw-native-sys/pypto-serving` | `0b0d8a06b682e079fbe5465498e1274017c26243` |

Enumeration metadata was also saved to:

- `tmp/restart-sources/hw-native-sys/repos.json`
- `tmp/restart-sources/hw-native-sys/repos.tsv`

## Additional Repository Sources

| Source | Destination | HEAD SHA |
| --- | --- | --- |
| `mirage-project/mirage` | `tmp/restart-sources/repos/mirage` | `2c87a75629e029492ae410ad976ca4a75012e551` |
| `vllm-project/vllm` | `tmp/restart-sources/repos/vllm` | `93bad119120d0f9bff707dcbf5af5c029158b969` |
| `flashinfer-ai/flashinfer` | `tmp/restart-sources/repos/flashinfer` | `9c5ed7c194e7412780862491742fc655daaad6ac` |

## Papers And Pages

| Source | Destination | Saved status |
| --- | --- | --- |
| `https://arxiv.org/abs/2604.13327` | `tmp/restart-sources/papers/2604.13327.html` | saved, 48477 bytes |
| `https://arxiv.org/pdf/2604.13327` | `tmp/restart-sources/papers/2604.13327.pdf` | saved, 1421924 bytes |
| `https://arxiv.org/abs/2601.19092` | `tmp/restart-sources/papers/2601.19092.html` | saved, 46321 bytes |
| `https://arxiv.org/pdf/2601.19092` | `tmp/restart-sources/papers/2601.19092.pdf` | saved, 1072795 bytes |
| Triton Python package install page | `tmp/restart-sources/pages/triton-installation-python-package.html` | saved, 10001 bytes |
| DeepSeek V4 self-hosting guide | `tmp/restart-sources/pages/deepseek-v4-self-hosting-guide.html` | saved, 141324 bytes |

The Triton page was downloaded from
`https://triton-lang.org/main/getting-started/installation.html#python-package`.
The DeepSeek V4 guide was downloaded from
`https://lushbinary.com/blog/deepseek-v4-self-hosting-guide-vllm-hardware-deployment/`.

## Failures Or Partial Acquisition

No source acquisition failures were observed. The acquisition log records
status 0 for every clone and download command. No partial source is known at
this inventory point.

## Non-Claims

This PR only inventories and records the restart source corpus. It does not
implement CUDA, run H200 kernels, load DeepSeek weights, start vLLM, or prove
serving output.

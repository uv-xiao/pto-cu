# NVIDIA Backend Source Manifest

This manifest records supplemental external source artifacts acquired for
`docs/in_progress/001-nvidia-backend.md`. The artifacts themselves live under
gitignored `tmp/sources/` and are not committed. The earlier clean-source
inventory in `source_inventory.md` records a second acquisition pass under
`tmp/restart-sources/`.

## Acquisition Summary

- Acquisition date: 2026-06-20.
- GitHub org query:
  `https://api.github.com/orgs/hw-native-sys/repos?per_page=100&type=all`
- GitHub org response:
  `tmp/sources/manifests/hw-native-sys-repos-page1.json`
- Git clone mode: shallow clone with `git clone --depth 1`.
- Paper/web download mode: `curl -fL`.

## hw-native-sys Repositories

- `https://github.com/hw-native-sys/pypto`
  - Local path: `tmp/sources/repos/hw-native-sys/pypto`
  - Branch: `main`
  - Commit: `2cd14c2cb076`
  - Why it matters: pypto target surface for simpler NVIDIA.
- `https://github.com/hw-native-sys/.github`
  - Local path: `tmp/sources/repos/hw-native-sys/.github`
  - Branch: `main`
  - Commit: `66d2246804ca`
  - Why it matters: organization workflow and metadata conventions.
- `https://github.com/hw-native-sys/simpler`
  - Local path: `tmp/sources/repos/hw-native-sys/simpler`
  - Branch: `main`
  - Commit: `cdbea2764f30`
  - Why it matters: upstream simpler runtime and compiler baseline.
- `https://github.com/hw-native-sys/PTOAS`
  - Local path: `tmp/sources/repos/hw-native-sys/PTOAS`
  - Branch: `main`
  - Commit: `da011a3d178e`
  - Why it matters: existing PTO assembly and tooling context.
- `https://github.com/hw-native-sys/pto-isa`
  - Local path: `tmp/sources/repos/hw-native-sys/pto-isa`
  - Branch: `main`
  - Commit: `e25732f09943`
  - Why it matters: Ascend ISA reference, not the NVIDIA compiler path.
- `https://github.com/hw-native-sys/pypto-lib`
  - Local path: `tmp/sources/repos/hw-native-sys/pypto-lib`
  - Branch: `main`
  - Commit: `1cf844e4859b`
  - Why it matters: pypto library features and compatibility context.
- `https://github.com/hw-native-sys/pypto_top_level_documents`
  - Local path: `tmp/sources/repos/hw-native-sys/pypto_top_level_documents`
  - Branch: `main`
  - Commit: `69fe802cc263`
  - Why it matters: project-level design and terminology references.
- `https://github.com/hw-native-sys/pypto-serving`
  - Local path: `tmp/sources/repos/hw-native-sys/pypto-serving`
  - Branch: `main`
  - Commit: `0b0d8a06b682`
  - Why it matters: candidate serving path for simpler NVIDIA kernels.

## External Implementation Repositories

- `https://github.com/triton-lang/triton`
  - Local path: `tmp/sources/repos/external/triton`
  - Branch: `main`
  - Commit: `ba1f2062e2c6`
  - Why it matters: Triton/Gluon generator and GPU codegen reference.
- `https://github.com/mirage-project/mirage`
  - Local path: `tmp/sources/repos/external/mirage`
  - Branch: `mpk`
  - Commit: `2c87a75629e0`
  - Why it matters: persistent-kernel and scheduling reference.
- `https://github.com/vllm-project/vllm`
  - Local path: `tmp/sources/repos/external/vllm`
  - Branch: `main`
  - Commit: `0a49fb2b13e4`
  - Why it matters: candidate serving integration and kernel-launch path.
- `https://github.com/flashinfer-ai/flashinfer`
  - Local path: `tmp/sources/repos/external/flashinfer`
  - Branch: `main`
  - Commit: `9c5ed7c194e7`
  - Why it matters: serving kernel reference for attention and decode paths.

## Codex Skill Sources

- `https://github.com/SihaoLiu/skills`
  - Local path: `tmp/sources/repos/skills/sihaoliu-skills`
  - Branch: `main`
  - Commit: `ec9cd9e2733f`
  - Why it matters: primary requested skill source for Codex adaptation.
- `https://github.com/PolyArch/humanize`
  - Local path: `tmp/sources/repos/skills/humanize`
  - Branch: `main`
  - Commit: `0ec921a36b43`
  - Why it matters: candidate supporting skill source.
- `https://github.com/obra/Superpowers`
  - Local path: `tmp/sources/repos/skills/superpowers`
  - Branch: `main`
  - Commit: `896224c4b187`
  - Why it matters: candidate supporting skill source.

## Distributed Communication Sources

- `https://github.com/ray-project/ray`
  - Local path: `tmp/sources/repos/communication/ray`
  - Branch: `master`
  - Commit: `51971fa737c3`
  - Why it matters: distributed orchestration and worker hierarchy reference.
- `https://github.com/NVIDIA/nccl`
  - Local path: `tmp/sources/repos/communication/nccl`
  - Branch: `master`
  - Commit: `5067397c2676`
  - Why it matters: NVIDIA collective and device communication reference.
- `https://github.com/uccl-project/uccl`
  - Local path: `tmp/sources/repos/communication/uccl`
  - Branch: `main`
  - Commit: `82e51a3ad944`
  - Why it matters: UCCL GPU communication and expert-parallel reference.

## Papers And Web Sources

- `https://arxiv.org/abs/2604.13327`
  - Local path: `tmp/sources/papers/arxiv-2604.13327.html`
  - Size: `48477 bytes`
  - Why it matters: requested source paper landing page.
- `https://arxiv.org/pdf/2604.13327`
  - Local path: `tmp/sources/papers/arxiv-2604.13327.pdf`
  - Size: `1421924 bytes`
  - Why it matters: requested source paper PDF.
- `https://arxiv.org/abs/2601.19092`
  - Local path: `tmp/sources/papers/arxiv-2601.19092.html`
  - Size: `45335 bytes`
  - Why it matters: requested source paper landing page.
- `https://arxiv.org/pdf/2601.19092`
  - Local path: `tmp/sources/papers/arxiv-2601.19092.pdf`
  - Size: `1072795 bytes`
  - Why it matters: requested source paper PDF.
- `https://arxiv.org/html/2604.19241v1`
  - Local path: `tmp/sources/papers/arxiv-2604.19241v1.html`
  - Size: `377916 bytes`
  - Why it matters: multi-GPU MoE dispatch/combine reference.
- Triton Python package install page
  - Local path: `tmp/sources/web/triton-installation-python-package.html`
  - Size: `10001 bytes`
  - Why it matters: Triton installation and package setup reference.
- DeepSeek-V4 self-hosting guide
  - Local path:
    `tmp/sources/web/deepseek-v4-self-hosting-guide-vllm-hardware-deployment.html`
  - Size: `141324 bytes`
  - Why it matters: DeepSeek-V4 serving hardware/deployment reference.
- `https://www.aihero.dev/my-grill-me-skill-has-gone-viral`
  - Local path: `tmp/sources/web/aihero-grill-me-skill.html`
  - Size: `145560 bytes`
  - Why it matters: requested grill-me skill reference.
- `https://uccl-project.github.io/`
  - Local path: `tmp/sources/web/uccl-project.html`
  - Size: `27970 bytes`
  - Why it matters: UCCL project overview and communication context.

## Follow-Up Sources

The source-acquisition pass covers the explicit repositories, papers, web
sources, Codex skill sources, and Ray/NCCL/UCCL communication references named
by the restart objective. Future child slices should add source rows only when
a new design decision cites a concrete additional project, paper, or
documentation page.

## Refresh Procedure

Refresh the manifest by re-running the GitHub org API query, fetching or
recloning the shallow repositories under `tmp/sources/repos/`, downloading the
paper/web files with `curl -fL`, and updating the commit and size fields above.
Do not commit the raw `tmp/sources/` contents.

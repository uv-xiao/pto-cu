# NVIDIA Backend Flow Details: Scope

## Scope

This is not a low-level compiler comparison. The important boundary is how a
`simpler` user gets from Python test/example code to prepared device work:

1. prebuilt runtime lookup or rebuild;
2. user callable compilation;
3. worker initialization;
4. callable preparation;
5. task launch;
6. tensor copy-back and teardown.


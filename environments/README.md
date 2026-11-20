# Tested environments

`pyproject.toml` declares minimum versions. The files in this directory record
two fully resolved environments in which the test suite and the offline safety
checks of release 0.1.0 pass.

| File | Python | PyTorch | Torchvision |
|---|---|---|---|
| `tested-python3.9.txt` | 3.9 | 2.8.0 | 0.23.0 |
| `tested-python3.10.txt` | 3.10 | 2.14.1 | 0.29.2 |

Use a file as a constraints file to obtain the same versions:

```bash
python -m pip install -e ".[dev]" -c environments/tested-python3.9.txt
```

The files were produced with `pip freeze` on Windows 11 from the default PyPI
wheels. On Linux, PyPI serves CUDA builds of PyTorch for the same version
numbers. To select a specific CUDA build, add the matching PyTorch index, for
example `--extra-index-url https://download.pytorch.org/whl/cu118`.

Checks that pass in both environments, without checkpoints, API keys, or
network access:

```bash
python -m pytest -q
python -m ophagent.reviewer_smoke
```

See [`docs/REPRODUCIBILITY.md`](../docs/REPRODUCIBILITY.md) for the relation
between these environments and the environment of the reported experiments.

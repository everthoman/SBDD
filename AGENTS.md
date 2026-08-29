# AGENTS.md

## Environment
- Project conda env: `sbdd` — run Python with `/home/evehom/miniconda3/envs/sbdd/bin/python` (base env lacks rdkit, rich-argparse, etc.).
- GNINA executable: `/opt/gnina/gnina` (default in the scripts).

## Hardware
- GPU 0 drives the display — never schedule compute work on it by default. `gnina.py` auto-selection skips display-driven cards (via nvidia-smi `display_active`); keep any new GPU logic consistent with that.

## Conventions
- Commit messages: lowercase, `scriptname: short description` (e.g. `gnina: make --num-gpus functional`).
- CLI scripts follow the house style: `rich_argparse` + `argcomplete`, `PYTHON_ARGCOMPLETE_OK` first line, subcommands where applicable.
- New scripts get a companion `<name>_README.md`.
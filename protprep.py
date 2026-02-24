#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
"""
Protein Preparation Script for Docking and Molecular Dynamics

Prepares protein structures for:
  - Molecular docking (GNINA)
  - Molecular dynamics and GB/SA simulations (OpenMM)

Pipeline:
  1. Clean     — chain selection, altloc resolution, water/HETATM removal
  2. Fix       — missing residue and heavy atom repair (PDBFixer)
  3. Protonate — hydrogen addition at target pH (PDB2PQR + PROPKA, or PDBFixer)
  4. Minimize  — vacuum energy minimization (OpenMM, optional)

Outputs:
  <name>_prepared.pdb   protonated structure for docking
  <name>_minimized.pdb  energy-minimized structure for MD/GB-SA  (if --minimize)

Dependencies:
  Required:     biopython, pdbfixer, openmm
  Protonation:  pdb2pqr  (pip install pdb2pqr)  — falls back to PDBFixer if absent
  UI:           rich_argparse, argcomplete        — optional, enhances CLI output

Written by Claude Sonnet 4.6, 2026-02-24
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

try:
    from rich_argparse import RawDescriptionRichHelpFormatter
except ImportError:
    from argparse import RawDescriptionHelpFormatter as RawDescriptionRichHelpFormatter  # type: ignore

try:
    import argcomplete
    from argcomplete.completers import FilesCompleter
    _ARGCOMPLETE = True
except ImportError:
    _ARGCOMPLETE = False

try:
    from Bio.PDB import PDBParser, PDBIO, Select
    _BIOPYTHON = True
except ImportError:
    _BIOPYTHON = False
    Select = object  # placeholder so the class definition below doesn't crash

try:
    from pdbfixer import PDBFixer
    from openmm.app import PDBFile
    _PDBFIXER = True
except ImportError:
    _PDBFIXER = False

try:
    import openmm
    import openmm.app as omm_app
    import openmm.unit as unit
    _OPENMM = True
except ImportError:
    _OPENMM = False


# ─── Utilities ────────────────────────────────────────────────────────────────

def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    elif seconds < 3600:
        return f"{seconds / 60:.2f} minutes"
    else:
        return f"{seconds / 3600:.2f} hours"


# ─── Fetch ────────────────────────────────────────────────────────────────────

def step_fetch(pdb_id, output_path):
    """Download a PDB file from RCSB."""
    pdb_id = pdb_id.upper()
    url = f'https://files.rcsb.org/download/{pdb_id}.pdb'
    print(f"  URL: {url}")
    try:
        urllib.request.urlretrieve(url, str(output_path))
    except urllib.error.HTTPError as e:
        print(f"[ERROR] Failed to fetch {pdb_id}: HTTP {e.code}")
        sys.exit(1)
    except urllib.error.URLError as e:
        print(f"[ERROR] Network error: {e.reason}")
        sys.exit(1)


# ─── BioPython helpers ────────────────────────────────────────────────────────

class _StructureSelector(Select):
    """Keep a single chain, resolve altlocs to A, remove water and unwanted HETATM."""

    def __init__(self, chain_id=None, keep_resnames=None):
        self.chain_id = chain_id
        self.keep_resnames = {r.strip().upper() for r in (keep_resnames or [])}

    def accept_chain(self, chain):
        return self.chain_id is None or chain.get_id() == self.chain_id

    def accept_residue(self, residue):
        hetflag, _, _ = residue.get_id()
        if hetflag == ' ':
            return True       # standard amino acid
        if hetflag == 'W':
            return False      # water
        return residue.get_resname().strip().upper() in self.keep_resnames

    def accept_atom(self, atom):
        return atom.get_altloc() in (' ', 'A')


def _inspect(pdb_path):
    parser = PDBParser(QUIET=True)
    s = parser.get_structure('p', str(pdb_path))[0]
    residues = list(s.get_residues())
    atoms    = list(s.get_atoms())
    return {
        'chains':    [c.get_id() for c in s.get_chains()],
        'n_std':     sum(1 for r in residues if r.get_id()[0] == ' '),
        'n_water':   sum(1 for r in residues if r.get_id()[0] == 'W'),
        'het_names': sorted({r.get_resname().strip() for r in residues
                             if r.get_id()[0] not in (' ', 'W')}),
        'n_altloc':  sum(1 for a in atoms if a.get_altloc() not in (' ', 'A')),
    }


# ─── Pipeline steps ───────────────────────────────────────────────────────────

def step_clean(input_pdb, output_pdb, chain_id=None, keep_het=None):
    """BioPython: chain selection, altloc resolution, HETATM/water removal."""
    if not _BIOPYTHON:
        print("[ERROR] BioPython is required: pip install biopython")
        sys.exit(1)

    info = _inspect(input_pdb)
    print(f"  Chains found:         {', '.join(info['chains'])}")
    print(f"  Standard residues:    {info['n_std']}")
    print(f"  Waters:               {info['n_water']}")
    if info['het_names']:
        print(f"  HETATM groups:        {', '.join(info['het_names'])}")
    if info['n_altloc']:
        print(f"  Altloc atoms:         {info['n_altloc']}  →  keeping A only")

    if chain_id and chain_id not in info['chains']:
        print(f"[ERROR] Chain '{chain_id}' not found. Available: {', '.join(info['chains'])}")
        sys.exit(1)

    parser = PDBParser(QUIET=True)
    struct = parser.get_structure('p', str(input_pdb))
    io = PDBIO()
    io.set_structure(struct)
    io.save(str(output_pdb), _StructureSelector(chain_id=chain_id, keep_resnames=keep_het))
    return info


def step_fix(input_pdb, output_pdb):
    """PDBFixer: fill missing residues and heavy atoms."""
    if not _PDBFIXER:
        print("[ERROR] PDBFixer is required: pip install pdbfixer")
        sys.exit(1)

    fixer = PDBFixer(filename=str(input_pdb))

    fixer.findMissingResidues()
    n_res = sum(len(v) for v in fixer.missingResidues.values())

    fixer.findNonstandardResidues()
    nonstandard = [(r.name, s) for r, s in fixer.nonstandardResidues]

    fixer.findMissingAtoms()
    n_atoms = sum(len(v) for v in fixer.missingAtoms.values())
    n_term  = sum(len(v) for v in fixer.missingTerminals.values())

    print(f"  Missing residues:     {n_res}")
    print(f"  Missing atoms:        {n_atoms}  (+ {n_term} terminal atoms)")
    if nonstandard:
        print(f"  Non-standard res:     {', '.join(f'{n} → {s}' for n, s in nonstandard)}")

    fixer.addMissingAtoms()

    with open(str(output_pdb), 'w') as fh:
        PDBFile.writeFile(fixer.topology, fixer.positions, fh)

    return {'n_missing_res': n_res, 'n_missing_atoms': n_atoms, 'nonstandard': nonstandard}


def step_protonate_pdb2pqr(input_pdb, output_pdb, ph, ff='AMBER'):
    """PDB2PQR + PROPKA: add hydrogens at target pH with AMBER atom naming."""
    pqr_out = output_pdb.with_suffix('.pqr')
    cmd = [
        'pdb2pqr',
        '--ff', ff,
        '--titration-state-method', 'propka',
        '--with-ph', str(ph),
        '--pdb-output', str(output_pdb),
        '--keep-chain',
        '--drop-water',
        str(input_pdb),
        str(pqr_out),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True,
                                cwd=str(output_pdb.parent))
    except FileNotFoundError:
        print("  [WARNING] pdb2pqr not found in PATH")
        return False
    if result.returncode != 0:
        print(f"  [WARNING] PDB2PQR exited with code {result.returncode}:")
        for line in result.stderr.strip().splitlines()[:8]:
            print(f"    {line}")
        return False

    for f in list(output_pdb.parent.glob('*.pqr')) + list(output_pdb.parent.glob('*.propka')):
        f.unlink(missing_ok=True)
    return True


def step_protonate_pdbfixer(input_pdb, output_pdb, ph):
    """PDBFixer fallback: add hydrogens at target pH without PROPKA."""
    if not _PDBFIXER:
        print("[ERROR] PDBFixer is required: pip install pdbfixer")
        sys.exit(1)
    fixer = PDBFixer(filename=str(input_pdb))
    fixer.addMissingHydrogens(ph)
    with open(str(output_pdb), 'w') as fh:
        PDBFile.writeFile(fixer.topology, fixer.positions, fh)


def step_minimize(fixed_pdb, output_pdb, ph, ff_xml='amber14-all.xml',
                  max_iter=1000, restraint_k=1000.0):
    """
    OpenMM constrained energy minimization with heavy-atom positional restraints.

    Loads the fixed PDB (no H) and adds hydrogens via OpenMM Modeller so that
    atom naming is guaranteed compatible with the force field.

    All non-hydrogen atoms are harmonically restrained to their starting positions
    with force constant restraint_k (kJ/mol/nm²). Hydrogens move freely.
    This preserves crystallographic heavy-atom coordinates while optimizing H
    geometry and resolving steric clashes introduced during structure repair.
    """
    if not _OPENMM:
        print("  [WARNING] OpenMM not found — skipping minimization.")
        print("            Install: conda install -c conda-forge openmm")
        return False

    pdb = omm_app.PDBFile(str(fixed_pdb))
    ff  = omm_app.ForceField(ff_xml)

    modeller = omm_app.Modeller(pdb.topology, pdb.positions)
    modeller.addHydrogens(ff, pH=ph)

    system = ff.createSystem(modeller.topology, nonbondedMethod=omm_app.NoCutoff)

    # Harmonic positional restraints on all non-hydrogen atoms.
    # Coordinates x, y, z are in nm; k is in kJ/mol/nm².
    restraint = openmm.CustomExternalForce(
        '0.5 * k * ((x - x0)^2 + (y - y0)^2 + (z - z0)^2)'
    )
    restraint.addGlobalParameter('k', restraint_k)
    restraint.addPerParticleParameter('x0')
    restraint.addPerParticleParameter('y0')
    restraint.addPerParticleParameter('z0')

    positions = modeller.positions
    n_restrained = 0
    for atom in modeller.topology.atoms():
        if atom.element is not None and atom.element.symbol != 'H':
            pos = positions[atom.index].value_in_unit(unit.nanometers)
            restraint.addParticle(atom.index, pos)
            n_restrained += 1
    system.addForce(restraint)
    print(f"  Heavy atoms restrained:{n_restrained}  (k = {restraint_k:.0f} kJ/mol/nm²)")

    integrator = openmm.LangevinMiddleIntegrator(
        300 * unit.kelvin,
        1  / unit.picosecond,
        0.004 * unit.picoseconds,
    )
    sim = omm_app.Simulation(modeller.topology, system, integrator)
    sim.context.setPositions(positions)

    e_before = (sim.context.getState(getEnergy=True)
                    .getPotentialEnergy()
                    .value_in_unit(unit.kilocalories_per_mole))
    print(f"  Energy before:        {e_before:>12.1f} kcal/mol")

    sim.minimizeEnergy(maxIterations=max_iter)

    state = sim.context.getState(getPositions=True, getEnergy=True)
    e_after = state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
    print(f"  Energy after:         {e_after:>12.1f} kcal/mol")

    with open(str(output_pdb), 'w') as fh:
        omm_app.PDBFile.writeFile(sim.topology, state.getPositions(), fh)
    return True


# ─── Argument parsing ─────────────────────────────────────────────────────────

def parse_args():
    desc = """
Protein preparation pipeline for docking and molecular dynamics.

Steps:
  1. Clean     — chain selection, altloc resolution, water/HETATM removal
  2. Fix       — missing residue and heavy atom repair (PDBFixer)
  3. Protonate — hydrogen addition at target pH (PDB2PQR + PROPKA, or PDBFixer)
  4. Minimize  — vacuum energy minimization (OpenMM, optional)

Outputs:
  <name>_prepared.pdb   protonated structure for docking
  <name>_minimized.pdb  energy-minimized structure for MD/GB-SA  (if --minimize)

Notes:
  - The prepared and minimized outputs use different hydrogen addition methods.
    prepared.pdb uses PDB2PQR/PROPKA (AMBER naming, optimal for docking).
    minimized.pdb uses OpenMM Modeller (force-field-native naming, for MD).
  - HETATM residues kept via --keep-het are passed through without
    parameterization; handle cofactor charges separately before MD.

Examples:
  %(prog)s -i protein.pdb
  %(prog)s -i protein.pdb --chain A --ph 6.5
  %(prog)s -i complex.pdb --chain A --keep-het HEM ZN --minimize
  %(prog)s -i apo.pdb --ph 7.4 --minimize --max-iter 2000
"""
    p = argparse.ArgumentParser(
        description=desc,
        formatter_class=RawDescriptionRichHelpFormatter,
    )

    io = p.add_argument_group('Input / Output')
    io.add_argument('--fetch', metavar='PDBID',
                    help='Fetch structure from RCSB PDB before preparing '
                         '(e.g. --fetch 1ABC). Sets --input to <PDBID>.pdb '
                         'if --input is not given.')
    inp = io.add_argument('-i', '--input', metavar='PDB',
                          help='Input PDB file (required unless --fetch is used)')
    io.add_argument('-o', '--output', metavar='PDB',
                    help='Output path for prepared structure '
                         '(default: <input>_prepared.pdb)')
    if _ARGCOMPLETE:
        inp.completer = FilesCompleter(allowednames=['.pdb', '.ent'])

    struct = p.add_argument_group('Structure')
    struct.add_argument('--chain', metavar='ID',
                        help='Extract a single chain (default: use all chains)')
    struct.add_argument('--keep-het', nargs='+', metavar='RESNAME', default=[],
                        help='HETATM residue names to keep, e.g. HEM ZN MG '
                             '(waters are always removed)')

    prot = p.add_argument_group('Protonation')
    prot.add_argument('--ph', type=float, default=7.4, metavar='FLOAT',
                      help='Target pH (default: 7.4)')
    prot.add_argument('--ff', default='AMBER',
                      choices=['AMBER', 'CHARMM', 'PARSE', 'TYL06'],
                      help='Force field for PDB2PQR atom naming (default: AMBER)')
    prot.add_argument('--no-pdb2pqr', action='store_true',
                      help='Use PDBFixer for hydrogen addition instead of PDB2PQR + PROPKA')

    mini = p.add_argument_group('Minimization')
    mini.add_argument('--minimize', action='store_true',
                      help='Run OpenMM vacuum energy minimization and write '
                           '<name>_minimized.pdb (for MD / GB-SA)')
    mini.add_argument('--max-iter', type=int, default=1000, metavar='N',
                      help='Maximum minimization iterations (default: 1000)')
    mini.add_argument('--restraint-k', type=float, default=1000.0, metavar='FLOAT',
                      help='Positional restraint force constant for heavy atoms '
                           'in kJ/mol/nm² (default: 1000.0). Higher values keep '
                           'heavy atoms closer to their starting positions.')

    pipe = p.add_argument_group('Pipeline')
    pipe.add_argument('--skip-fix', action='store_true',
                      help='Skip PDBFixer step (use when structure has no missing atoms)')
    pipe.add_argument('--keep-intermediates', action='store_true',
                      help='Save intermediate PDB files from each pipeline step')

    if _ARGCOMPLETE:
        argcomplete.autocomplete(p)

    return p.parse_args()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    t0 = time.time()

    if not args.input and not args.fetch:
        print("[ERROR] Provide --input or --fetch PDBID")
        sys.exit(1)

    if args.fetch and not args.input:
        args.input = f'{args.fetch.upper()}.pdb'

    input_pdb = Path(args.input).resolve()

    if args.fetch:
        print(f"\n[FETCH] Downloading {args.fetch.upper()} from RCSB PDB...")
        step_fetch(args.fetch, input_pdb)
        print(f"[INFO] Saved  →  {input_pdb.name}")
    elif not input_pdb.exists():
        print(f"[ERROR] File not found: {input_pdb}")
        sys.exit(1)

    prepared_pdb = (Path(args.output).resolve() if args.output
                    else input_pdb.with_name(input_pdb.stem + '_prepared.pdb'))
    stem = prepared_pdb.stem.removesuffix('_prepared')
    minimized_pdb = prepared_pdb.with_name(stem + '_minimized.pdb')

    n_steps = 3 + int(args.minimize)
    prot_method = 'PDBFixer' if args.no_pdb2pqr else 'PDB2PQR + PROPKA'

    print()
    if args.fetch:
        print(f"[CONFIG] Fetch:        {args.fetch.upper()} (RCSB PDB)")
    print(f"[CONFIG] Input:        {input_pdb}")
    print(f"[CONFIG] Output:       {prepared_pdb}")
    if args.minimize:
        print(f"[CONFIG] Minimized:    {minimized_pdb}")
    print(f"[CONFIG] Chain:        {args.chain or 'all'}")
    print(f"[CONFIG] Keep HETATM:  {', '.join(args.keep_het) or 'none'}")
    print(f"[CONFIG] pH:           {args.ph}")
    print(f"[CONFIG] Protonation:  {prot_method}")
    print(f"[CONFIG] Fix missing:  {'no' if args.skip_fix else 'yes'}")
    print(f"[CONFIG] Minimize:     {'yes' if args.minimize else 'no'}"
          + (f"  (restraint k = {args.restraint_k:.0f} kJ/mol/nm²)" if args.minimize else ""))

    stats = {}

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        # ── Step 1: Clean ────────────────────────────────────────────────────
        print(f"\n[STEP 1/{n_steps}] Cleaning structure...")
        clean_pdb = tmp / 'clean.pdb'
        info = step_clean(input_pdb, clean_pdb,
                          chain_id=args.chain, keep_het=args.keep_het)
        stats['info'] = info
        if args.keep_intermediates:
            dest = prepared_pdb.with_name(prepared_pdb.stem + '_s1_clean.pdb')
            shutil.copy(clean_pdb, dest)
            print(f"  Saved: {dest.name}")
        print(f"[INFO] Cleaning done.")

        # ── Step 2: Fix ──────────────────────────────────────────────────────
        if not args.skip_fix:
            print(f"\n[STEP 2/{n_steps}] Repairing missing residues and atoms...")
            fixed_pdb = tmp / 'fixed.pdb'
            fix_stats = step_fix(clean_pdb, fixed_pdb)
            stats['fix'] = fix_stats
            if args.keep_intermediates:
                dest = prepared_pdb.with_name(prepared_pdb.stem + '_s2_fixed.pdb')
                shutil.copy(fixed_pdb, dest)
                print(f"  Saved: {dest.name}")
            print(f"[INFO] Repair done.")
        else:
            fixed_pdb = clean_pdb
            print(f"\n[STEP 2/{n_steps}] Skipping repair (--skip-fix).")

        # ── Step 3: Protonate ────────────────────────────────────────────────
        print(f"\n[STEP 3/{n_steps}] Protonating at pH {args.ph}...")
        prot_pdb = tmp / 'protonated.pdb'
        if args.no_pdb2pqr:
            step_protonate_pdbfixer(fixed_pdb, prot_pdb, args.ph)
            stats['prot'] = 'pdbfixer'
        else:
            ok = step_protonate_pdb2pqr(fixed_pdb, prot_pdb, args.ph, args.ff)
            if ok:
                stats['prot'] = 'pdb2pqr+propka'
            else:
                print(f"  Falling back to PDBFixer hydrogen addition...")
                step_protonate_pdbfixer(fixed_pdb, prot_pdb, args.ph)
                stats['prot'] = 'pdbfixer_fallback'
        shutil.copy(prot_pdb, prepared_pdb)
        print(f"[INFO] Protonation done  →  {prepared_pdb.name}")

        # ── Step 4: Minimize ─────────────────────────────────────────────────
        if args.minimize:
            print(f"\n[STEP 4/{n_steps}] Energy minimization (OpenMM ff14SB, vacuum)...")
            ok = step_minimize(fixed_pdb, minimized_pdb, ph=args.ph,
                               max_iter=args.max_iter, restraint_k=args.restraint_k)
            if ok:
                print(f"[INFO] Minimization done →  {minimized_pdb.name}")

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    bar = '─' * 60
    i = stats.get('info', {})
    f = stats.get('fix', {})
    prot_labels = {
        'pdb2pqr+propka':    'PDB2PQR + PROPKA',
        'pdbfixer':          'PDBFixer',
        'pdbfixer_fallback': 'PDBFixer (PDB2PQR failed)',
    }
    removed_het = [h for h in i.get('het_names', [])
                   if h not in {r.upper() for r in args.keep_het}]

    print(f"\n{bar}")
    print(f"[SUMMARY] Input:              {input_pdb.name}")
    print(f"[SUMMARY] Chains processed:   {', '.join(i.get('chains', ['?']))}")
    print(f"[SUMMARY] Waters removed:     {i.get('n_water', '?')}")
    if removed_het:
        print(f"[SUMMARY] HETATM removed:     {', '.join(removed_het)}")
    if args.keep_het:
        print(f"[SUMMARY] HETATM kept:        {', '.join(args.keep_het)}")
    if i.get('n_altloc'):
        print(f"[SUMMARY] Altlocs resolved:   {i['n_altloc']} atoms")
    if f:
        print(f"[SUMMARY] Missing res added:  {f.get('n_missing_res', 0)}")
        print(f"[SUMMARY] Missing atoms added:{f.get('n_missing_atoms', 0)}")
        if f.get('nonstandard'):
            print(f"[SUMMARY] Non-std residues:   "
                  f"{', '.join(n for n, _ in f['nonstandard'])}")
    print(f"[SUMMARY] Protonation:        "
          f"{prot_labels.get(stats.get('prot', ''), '?')}  (pH {args.ph})")
    print(f"[SUMMARY] Output:             {prepared_pdb}")
    if args.minimize and minimized_pdb.exists():
        print(f"[SUMMARY] Minimized:          {minimized_pdb}")
    print(f"[SUMMARY] Elapsed:            {format_time(elapsed)}")
    print(f"{bar}\n")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
"""
MM-GB/SA Binding Free Energy Calculator

Estimates protein–ligand binding free energy using the end-point MM-GB/SA method:

  ΔGbind ≈ G(complex) − G(protein) − G(ligand)

Each G is the OpenMM potential energy under an implicit generalised-Born solvent
model, which includes:
  • Molecular mechanics energy (bonds, angles, dihedrals, vdW, Coulomb) in vacuo
  • Generalised Born electrostatic solvation free energy
  • Non-polar solvation free energy (SASA-based surface area term)

Entropy contributions are NOT included — the result is an enthalpic estimate
well suited for ranking docking poses or triaging virtual-screening hits.

Workflow per pose (single-trajectory MM-GB/SA):
  1. Build protein–ligand complex topology with OpenMM Modeller
  2. Optionally minimise the complex to relieve steric clashes
  3. Extract minimised positions for complex, protein, and ligand sub-sets
  4. Evaluate G for each sub-system at the (minimised) geometry
  5. ΔGbind = G_complex − G_protein − G_ligand

Force fields:
  Protein  → AMBER ff14SB    (amber14-all.xml)
  Ligand   → GAFF2           (openmmforcefields, AM1-BCC charges)
  Solvent  → GBn2 by default (most accurate AMBER implicit solvent model)

Inputs:
  -p / --protein   Prepared protein PDB (from protprep.py, ideally --minimize)
  -l / --ligands   Docking poses SDF   (from gnina_*.py or similar tools)

Output: CSV with ΔGbind and energy components for every pose, sorted by ΔGbind.

Dependencies:
  Required:  openmm  openmmforcefields  openff-toolkit  rdkit
  AM1-BCC:   ambertools (antechamber)  OR  openeye-toolkits
             (openmmforcefields falls back to RDKit-based charges if absent)
  Optional:  rich_argparse  argcomplete  — enhanced CLI output

Written by Claude Sonnet 4.6, 2026-02-24
"""

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Optional

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
    import numpy as np
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
    _RDKIT = True
except ImportError:
    _RDKIT = False

try:
    import openmm
    import openmm.app as app
    import openmm.unit as unit
    _OPENMM = True
except ImportError:
    _OPENMM = False

try:
    from openmmforcefields.generators import GAFFTemplateGenerator
    _OPENMMFF = True
except ImportError:
    _OPENMMFF = False

try:
    try:
        from openff.toolkit import Molecule as OFFMolecule
    except ImportError:
        from openff.toolkit.topology import Molecule as OFFMolecule
    _OPENFF = True
except ImportError:
    _OPENFF = False


# ─── Constants ────────────────────────────────────────────────────────────────

_KCAL = unit.kilocalories_per_mole if _OPENMM else None
_NM   = unit.nanometers             if _OPENMM else None

GB_MODELS: dict = {}
if _OPENMM:
    GB_MODELS = {
        'HCT':  app.HCT,
        'OBC1': app.OBC1,
        'OBC2': app.OBC2,
        'GBn':  app.GBn,
        'GBn2': app.GBn2,
    }


# ─── Utilities ────────────────────────────────────────────────────────────────

def format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


def check_dependencies() -> None:
    missing = []
    if not _RDKIT:    missing.append('rdkit')
    if not _OPENMM:   missing.append('openmm')
    if not _OPENMMFF: missing.append('openmmforcefields')
    if not _OPENFF:   missing.append('openff-toolkit')
    if missing:
        print(f"[ERROR] Missing required packages: {', '.join(missing)}")
        print("        Install via conda:")
        print(f"          conda install -c conda-forge {' '.join(missing)}")
        sys.exit(1)


def get_pose_name(mol, id_col: Optional[str], index: int) -> str:
    """Return the pose identifier from SDF properties or a numeric fallback."""
    if id_col and mol.HasProp(id_col):
        return mol.GetProp(id_col).strip()
    name = mol.GetProp('_Name').strip() if mol.HasProp('_Name') else ''
    return name if name else f"pose_{index}"


# ─── Ligand helpers ───────────────────────────────────────────────────────────

def ensure_hydrogens(rdkit_mol):
    """
    Return a molecule guaranteed to have explicit 3D hydrogens.

    If the input already has Hs, it is returned unchanged.
    Otherwise Hs are added with 3D coordinates derived from the existing heavy-
    atom geometry (Chem.AddHs with addCoords=True).

    Returns (mol_with_H, was_modified).
    """
    if rdkit_mol.GetNumAtoms() > Chem.RemoveHs(rdkit_mol).GetNumAtoms():
        return rdkit_mol, False
    return Chem.AddHs(rdkit_mol, addCoords=True), True


def rdkit_to_off(rdkit_mol):
    """Convert an RDKit molecule (with explicit Hs) to an OpenFF Molecule."""
    return OFFMolecule.from_rdkit(rdkit_mol, allow_undefined_stereo=True)


def off_to_omm_topology(off_mol):
    """Return an OpenMM Topology for a small molecule from its OpenFF Molecule."""
    return off_mol.to_topology().to_openmm()


def get_omm_positions(rdkit_mol):
    """
    Extract 3D coordinates from an RDKit conformer as an OpenMM Quantity (nm).
    """
    pos_ang = np.array(rdkit_mol.GetConformer().GetPositions())  # Å
    return unit.Quantity(pos_ang / 10.0, _NM)


# ─── OpenMM helpers ───────────────────────────────────────────────────────────

def build_forcefield(off_molecules: list, protein_ff: str, ligand_ff: str):
    """
    Build a combined AMBER + GAFF2 ForceField object.

    off_molecules : list of unique OFFMolecule instances to register with GAFF.
    """
    gaff = GAFFTemplateGenerator(molecules=off_molecules, forcefield=ligand_ff)
    ff   = app.ForceField(protein_ff)
    ff.registerTemplateGenerator(gaff.generator)
    return ff


def _create_system(topology, forcefield, gb_model, salt_conc_molar: float):
    """Create an OpenMM System with NoCutoff implicit GB solvent."""
    return forcefield.createSystem(
        topology,
        nonbondedMethod=app.NoCutoff,
        implicitSolvent=gb_model,
        implicitSolventSaltConc=salt_conc_molar * unit.molar,
        constraints=None,
        removeCMMotion=False,
    )


def _best_platform():
    """Return the fastest available OpenMM Platform (CUDA > OpenCL > CPU)."""
    for name in ('CUDA', 'OpenCL', 'CPU'):
        try:
            return openmm.Platform.getPlatformByName(name)
        except Exception:
            continue
    return None


def _make_simulation(topology, system):
    """Instantiate a Simulation with a simple Verlet integrator."""
    integrator = openmm.VerletIntegrator(1.0 * unit.femtoseconds)
    platform   = _best_platform()
    return (app.Simulation(topology, system, integrator, platform)
            if platform else
            app.Simulation(topology, system, integrator))


def minimize_system(topology, positions, forcefield, gb_model,
                    salt_conc: float, max_iter: int):
    """
    Energy-minimise a system and return (pos_array_nm, energy_kcal).
    pos_array_nm is a numpy array of shape (n_atoms, 3) in nanometres.
    """
    system = _create_system(topology, forcefield, gb_model, salt_conc)
    sim    = _make_simulation(topology, system)
    sim.context.setPositions(positions)
    sim.minimizeEnergy(maxIterations=max_iter)
    state  = sim.context.getState(getPositions=True, getEnergy=True)
    pos_nm = state.getPositions(asNumpy=True).value_in_unit(_NM)
    energy = state.getPotentialEnergy().value_in_unit(_KCAL)
    return pos_nm, energy


def energy_at(topology, positions, forcefield, gb_model, salt_conc: float) -> float:
    """Evaluate and return potential energy (kcal/mol) at given positions."""
    system = _create_system(topology, forcefield, gb_model, salt_conc)
    sim    = _make_simulation(topology, system)
    sim.context.setPositions(positions)
    state  = sim.context.getState(getEnergy=True)
    return state.getPotentialEnergy().value_in_unit(_KCAL)


# ─── MM-GB/SA core ────────────────────────────────────────────────────────────

def compute_mmgbsa(
    protein_top,
    protein_pos,
    lig_top,
    lig_rdkit_mol,
    forcefield,
    gb_model,
    salt_conc: float,
    do_minimize: bool,
    max_iter: int,
) -> dict:
    """
    Compute MM-GB/SA ΔGbind for a single protein–ligand pose.

    Single-trajectory approach:
      • Build complex topology (protein + ligand)
      • Optionally minimise the complex
      • Extract protein and ligand positions from the (minimised) complex
      • Evaluate G for each sub-system at those positions
      • ΔGbind = G_complex − G_protein − G_ligand

    Returns a dict with keys: dG_bind, E_complex, E_protein, E_ligand (kcal/mol).
    """
    n_prot  = protein_top.getNumAtoms()
    lig_pos = get_omm_positions(lig_rdkit_mol)

    # Build complex topology and merged positions
    modeller = app.Modeller(protein_top, protein_pos)
    modeller.add(lig_top, lig_pos)
    complex_top = modeller.topology
    complex_pos = modeller.positions

    if do_minimize:
        complex_nm, E_complex = minimize_system(
            complex_top, complex_pos, forcefield, gb_model, salt_conc, max_iter
        )
        complex_pos  = unit.Quantity(complex_nm,          _NM)
        prot_pos_use = unit.Quantity(complex_nm[:n_prot], _NM)
        lig_pos_use  = unit.Quantity(complex_nm[n_prot:], _NM)
    else:
        E_complex    = energy_at(complex_top, complex_pos, forcefield, gb_model, salt_conc)
        prot_pos_use = protein_pos
        lig_pos_use  = lig_pos

    E_protein = energy_at(protein_top, prot_pos_use, forcefield, gb_model, salt_conc)
    E_ligand  = energy_at(lig_top,     lig_pos_use,  forcefield, gb_model, salt_conc)

    return {
        'dG_bind':   E_complex - E_protein - E_ligand,
        'E_complex': E_complex,
        'E_protein': E_protein,
        'E_ligand':  E_ligand,
    }


# ─── Argument parsing ─────────────────────────────────────────────────────────

def parse_args():
    desc = """
MM-GB/SA binding free energy estimator.

For each pose in POSES_SDF, computes:
  ΔGbind ≈ G_complex − G_protein − G_ligand  (kcal/mol)

using AMBER ff14SB + GAFF2 + GBn2 implicit solvent (defaults).
Results are written to a CSV sorted by ΔGbind (most negative = tightest).

Examples:
  %(prog)s -p receptor_minimized.pdb -l docking_poses.sdf
  %(prog)s -p receptor.pdb -l poses.sdf -o results.csv --no-minimize
  %(prog)s -p receptor.pdb -l poses.sdf --gb-model OBC2 --salt-conc 0.1
  %(prog)s -p receptor.pdb -l poses.sdf --id-col "Compound_ID" --max-iter 1000
"""
    p = argparse.ArgumentParser(
        description=desc,
        formatter_class=RawDescriptionRichHelpFormatter,
    )

    io = p.add_argument_group('Input / Output')
    prot_arg = io.add_argument(
        '-p', '--protein', required=True, metavar='PDB',
        help='Prepared protein PDB file (from protprep.py, ideally with --minimize)',
    )
    lig_arg = io.add_argument(
        '-l', '--ligands', required=True, metavar='SDF',
        help='Docking poses SDF file (one or more entries)',
    )
    io.add_argument(
        '-o', '--output', metavar='CSV',
        help='Output CSV path (default: <ligands>_mmgbsa.csv)',
    )
    io.add_argument(
        '--id-col', metavar='PROP', default=None,
        help='SDF property to use as pose name (default: molecule title)',
    )

    ff = p.add_argument_group('Force Field / Solvent')
    ff.add_argument(
        '--protein-ff', default='amber14-all.xml', metavar='XML',
        help='AMBER protein force-field XML (default: amber14-all.xml)',
    )
    ff.add_argument(
        '--ligand-ff', default='gaff-2.11', metavar='FF',
        help='GAFF force-field version for ligands (default: gaff-2.11)',
    )
    ff.add_argument(
        '--gb-model', default='GBn2',
        choices=['HCT', 'OBC1', 'OBC2', 'GBn', 'GBn2'],
        help='Implicit solvent GB model (default: GBn2)',
    )
    ff.add_argument(
        '--salt-conc', type=float, default=0.15, metavar='M',
        help='Implicit solvent salt concentration mol/L (default: 0.15)',
    )

    opt = p.add_argument_group('Minimization')
    opt.add_argument(
        '--no-minimize', action='store_true',
        help='Skip energy minimization — use raw docked coordinates',
    )
    opt.add_argument(
        '--max-iter', type=int, default=500, metavar='N',
        help='Max minimization iterations per pose (default: 500)',
    )

    if _ARGCOMPLETE:
        prot_arg.completer = FilesCompleter(allowednames=['.pdb'])
        lig_arg.completer  = FilesCompleter(allowednames=['.sdf'])
        argcomplete.autocomplete(p)

    return p.parse_args()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    check_dependencies()
    t0 = time.time()

    protein_pdb = Path(args.protein)
    ligands_sdf = Path(args.ligands)

    if not protein_pdb.exists():
        print(f"[ERROR] Protein file not found: {protein_pdb}")
        sys.exit(1)
    if not ligands_sdf.exists():
        print(f"[ERROR] Ligands file not found: {ligands_sdf}")
        sys.exit(1)

    output_csv  = (Path(args.output) if args.output
                   else ligands_sdf.with_name(ligands_sdf.stem + '_mmgbsa.csv'))
    gb_model    = GB_MODELS[args.gb_model]
    do_minimize = not args.no_minimize

    print()
    print(f"[CONFIG] Protein:      {protein_pdb}")
    print(f"[CONFIG] Ligands:      {ligands_sdf}")
    print(f"[CONFIG] Output:       {output_csv}")
    print(f"[CONFIG] GB model:     {args.gb_model}")
    print(f"[CONFIG] Salt conc:    {args.salt_conc} M")
    print(f"[CONFIG] Protein FF:   {args.protein_ff}")
    print(f"[CONFIG] Ligand FF:    {args.ligand_ff}")
    print(f"[CONFIG] Minimize:     "
          f"{'no' if args.no_minimize else f'yes  (max {args.max_iter} iter)'}")

    # ── Load protein ──────────────────────────────────────────────────────────
    print("\n[INFO] Loading protein...")
    pdb          = app.PDBFile(str(protein_pdb))
    protein_top  = pdb.topology
    protein_pos  = pdb.positions
    n_prot_res   = sum(1 for _ in protein_top.residues())
    n_prot_atoms = protein_top.getNumAtoms()
    print(f"       {n_prot_res} residues, {n_prot_atoms} atoms")

    # ── Load all poses ────────────────────────────────────────────────────────
    print("\n[INFO] Loading ligand poses...")
    suppl        = Chem.SDMolSupplier(str(ligands_sdf), removeHs=False, sanitize=True)
    all_mols     = [m for m in suppl if m is not None]
    if not all_mols:
        print("[ERROR] No valid molecules found in the SDF file.")
        sys.exit(1)
    print(f"       {len(all_mols)} pose(s) loaded")

    # ── Ensure Hs; collect unique OFFMolecules for parameterisation ───────────
    print("\n[INFO] Preparing ligands for parameterisation...")
    processed    = []   # list of (mol_with_H, canonical_smiles)
    smi_to_info  = {}   # smiles → (off_mol, omm_topology)  or  None on failure
    n_added_H    = 0

    for mol in all_mols:
        mol_H, added = ensure_hydrogens(mol)
        if added:
            n_added_H += 1

        smi = Chem.MolToSmiles(Chem.RemoveHs(mol_H))

        if smi not in smi_to_info:
            try:
                off_mol = rdkit_to_off(mol_H)
                lig_top = off_to_omm_topology(off_mol)
                smi_to_info[smi] = (off_mol, lig_top)
            except Exception as e:
                print(f"  [WARNING] Could not parameterise {smi[:50]}: {e}")
                smi_to_info[smi] = None

        processed.append((mol_H, smi))

    unique_off_mols = [v[0] for v in smi_to_info.values() if v is not None]
    n_unique        = len(unique_off_mols)
    n_failed_prep   = sum(1 for v in smi_to_info.values() if v is None)

    print(f"       {len(all_mols)} pose(s), {n_unique} unique topology/topologies")
    if n_added_H:
        print(f"       [NOTE] Explicit Hs were added to {n_added_H} pose(s). "
              "Prefer SDF files with pre-added hydrogens for accurate H positions.")
    if n_failed_prep:
        print(f"       [WARNING] {n_failed_prep} unique topology/topologies "
              "could not be parameterised and will be skipped.")

    if not unique_off_mols:
        print("[ERROR] No ligands could be parameterised. Aborting.")
        sys.exit(1)

    # ── Build combined force field (AM1-BCC charges computed here) ────────────
    print(f"\n[INFO] Building force field ({args.protein_ff} + {args.ligand_ff})...")
    print("       Computing AM1-BCC charges for unique ligand(s) — may take a moment...")
    try:
        forcefield = build_forcefield(unique_off_mols, args.protein_ff, args.ligand_ff)
    except Exception as e:
        print(f"[ERROR] Force-field setup failed: {e}")
        print("        Make sure antechamber (AmberTools) or openeye-toolkits is")
        print("        installed for AM1-BCC charge calculation.")
        sys.exit(1)
    print("       Force field ready.")

    # ── MM-GB/SA per pose ─────────────────────────────────────────────────────
    results = []
    n_ok    = 0
    n_fail  = 0

    print(f"\n[INFO] Running MM-GB/SA for {len(processed)} pose(s)...")

    for i, (mol_H, smi) in enumerate(processed):
        pose_name = get_pose_name(all_mols[i], args.id_col, i + 1)
        t_pose    = time.time()
        info      = smi_to_info.get(smi)

        if info is None:
            print(f"  [{i+1}/{len(processed)}] {pose_name}  →  SKIP (parameterisation failed)")
            results.append({
                'pose': pose_name, 'dG_bind': '', 'E_complex': '',
                'E_protein': '', 'E_ligand': '', 'status': 'FAILED (parameterisation)',
            })
            n_fail += 1
            continue

        _, lig_top = info

        try:
            en = compute_mmgbsa(
                protein_top, protein_pos,
                lig_top, mol_H,
                forcefield, gb_model,
                args.salt_conc, do_minimize, args.max_iter,
            )
            dt = time.time() - t_pose
            print(f"  [{i+1}/{len(processed)}] {pose_name:<35} "
                  f"ΔGbind = {en['dG_bind']:>9.2f} kcal/mol  ({format_time(dt)})")
            results.append({
                'pose':      pose_name,
                'dG_bind':   f"{en['dG_bind']:.4f}",
                'E_complex': f"{en['E_complex']:.4f}",
                'E_protein': f"{en['E_protein']:.4f}",
                'E_ligand':  f"{en['E_ligand']:.4f}",
                'status':    'OK',
            })
            n_ok += 1
        except Exception as e:
            dt = time.time() - t_pose
            print(f"  [{i+1}/{len(processed)}] {pose_name}  →  FAILED: {e}")
            results.append({
                'pose': pose_name, 'dG_bind': '', 'E_complex': '',
                'E_protein': '', 'E_ligand': '', 'status': f'FAILED: {e}',
            })
            n_fail += 1

    # ── Sort and write CSV ────────────────────────────────────────────────────
    ok_rows   = [r for r in results if r['status'] == 'OK']
    fail_rows = [r for r in results if r['status'] != 'OK']
    ok_rows.sort(key=lambda r: float(r['dG_bind']))

    fieldnames = ['pose', 'dG_bind', 'E_complex', 'E_protein', 'E_ligand', 'status']
    with open(output_csv, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(ok_rows + fail_rows)

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    bar = '─' * 60
    print(f"\n{bar}")
    print(f"[SUMMARY] Poses:             {len(processed)}")
    print(f"[SUMMARY] Successful:        {n_ok}")
    if n_fail:
        print(f"[SUMMARY] Failed:            {n_fail}")
    if ok_rows:
        best  = ok_rows[0]
        worst = ok_rows[-1]
        print(f"[SUMMARY] Best  ΔGbind:      {float(best['dG_bind']):>8.2f} kcal/mol  ({best['pose']})")
        print(f"[SUMMARY] Worst ΔGbind:      {float(worst['dG_bind']):>8.2f} kcal/mol  ({worst['pose']})")
    print(f"[SUMMARY] Output:            {output_csv}")
    print(f"[SUMMARY] Elapsed:           {format_time(elapsed)}")
    print(f"{bar}\n")


if __name__ == '__main__':
    main()

# Physics Tripwires

**An independent oracle for AI agent driven atomistic simulation.**

Coding agents now write and run complete atomistic simulations, often with machine learned interatomic potentials as the engine. Both layers can fail silently: the job finishes, the numbers look plausible, and the physics is wrong. Such errors are usually checked by another AI reading the code, so the code and its checks share an author.

Physics Tripwires is a library of independent tests built from symmetry, conservation laws and consistency relations. Any correct atomistic workflow must satisfy them, whoever wrote it, and the pattern of failures helps show whether the script or the potential is at fault.

This repository holds the working pilot.

## The checks

- **Invariance:** translation, rotation and permutation of atoms
- **Force and energy consistency:** forces equal minus the gradient of the energy
- **Closed loop work:** zero net work around a closed path
- **Supercell extensivity:** energy per atom does not depend on cell size
- **Equipartition** at velocity initialisation
- **NVE energy conservation** during molecular dynamics

They work with any ASE calculator, for example EMT, MACE, UMA, GRACE, VASP or LAMMPS.

## Run the pilot

```bash
pip install ase
cd pilot
python run_pilot.py
```

The pilot seeds ten realistic silent errors into a 108 atom Cu workflow (ASE EMT engine) and compares typical agent style self tests with the tripwires. It takes about one minute on one CPU core.

| Seeded silent error | Agent style tests | Tripwires |
|---|---|---|
| Time step written as 2 instead of 2 fs | missed | caught |
| Temperature given in eV instead of K | missed | caught |
| Forces in kcal/mol with energy in eV | missed | caught |
| Force sign slip | missed | caught |
| Per atom energy reported as total energy | missed | caught |
| Leftover external field term | caught | caught |
| Periodic boundaries switched off | caught | caught |
| Non conservative (direct force) potential | caught | caught |
| Noisy forces (unconverged SCF) | caught | caught |
| Time step of 8 fs for Cu (energy still conserved) | missed | missed |
| **Total** | **4 / 10** | **9 / 10** |

The correct reference workflow passes all tripwires (no false alarms).

## Related work by the author

- [symlint](https://github.com/Asif-Iqbal-Bhatti/symlint): a symmetry based physics guardrail for AI driven materials discovery
- [vasp-grace-tensorpotential](https://github.com/Asif-Iqbal-Bhatti/vasp-grace-tensorpotential): run VASP inputs on GRACE machine learned potentials

## License

MIT

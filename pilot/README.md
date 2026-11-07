# Physics Tripwires (pilot)

Independent, physics derived checks for atomistic simulation workflows, including workflows written by AI coding agents.

The checks test relations that must hold whoever wrote the code: translation, rotation and permutation invariance, force and energy consistency, closed loop work, supercell extensivity, equipartition at initialisation, and NVE energy conservation. They run on any ASE calculator (EMT, MACE, UMA, GRACE, VASP, LAMMPS).

## Run the pilot

```bash
pip install ase
python run_pilot.py
```

The pilot seeds ten realistic silent errors into a Cu workflow (EMT engine, 108 atoms) and compares typical agent style self tests with the tripwires. It takes about one minute on one CPU core.

Result: agent style tests catch 4 of 10, tripwires catch 9 of 10, with no false alarm on the correct workflow.

## License

MIT

"""Pilot: seed ten realistic silent errors into a Cu workflow and compare two test suites:
(a) the kind of self tests a coding agent typically writes, and (b) Physics Tripwires.
Engine: ASE EMT (runs on a laptop in about a minute). Swap in any ASE calculator (MACE, UMA, VASP).
"""
import json
import numpy as np
from ase import units
from ase.build import bulk
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.emt import EMT
from ase.eos import EquationOfState

import tripwires as tw

A0 = 3.59  # EMT equilibrium lattice constant of Cu (approx.)


class Wrap(Calculator):
    """A calculator wrapper that injects one silent error into energies or forces."""
    implemented_properties = ["energy", "forces"]

    def __init__(self, mode=None):
        super().__init__()
        self.mode = mode
        self.base = EMT()

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        a = atoms.copy()
        a.calc = self.base
        e = a.get_potential_energy()
        f = a.get_forces()
        m = self.mode
        if m == "force_kcal":            # forces converted to kcal/mol/A, energy left in eV
            f = f * 23.0605
        elif m == "force_sign":          # sign slip when adapting gradient output
            f = -f
        elif m == "per_atom_energy":     # per atom energy returned as total energy
            e = e / len(a)
        elif m == "field_term":          # leftover external field / dipole correction term
            e = e + 0.02 * a.positions[:, 2].sum()
            f = f.copy()
            f[:, 2] -= 0.02
        elif m == "nonconservative":     # direct force model: small rotational component
            c = a.positions.mean(axis=0)
            r = a.positions - c
            f = f + 0.03 * np.stack([-r[:, 1], r[:, 0], 0 * r[:, 0]], axis=1)
        elif m == "noisy_forces":        # unconverged SCF / low precision inference
            f = f + np.random.default_rng().normal(0, 0.05, f.shape)
        self.results = {"energy": e, "forces": f}


def make_cell(pbc=True):
    def mk():
        a = bulk("Cu", "fcc", a=A0, cubic=True) * (3, 3, 3)
        a.pbc = pbc
        return a
    return mk


def workflow(calc_mode=None, dt=2 * units.fs, temperature_arg=300, pbc=True):
    return dict(calc=Wrap(calc_mode), make_atoms=make_cell(pbc), dt=dt,
                temperature_arg=temperature_arg, target_K=300, element="Cu", a0=A0, pbc=pbc)


CASES = {
    "reference (correct)":                     workflow(),
    "S1 time step without units (2 not 2 fs)": workflow(dt=2.0),
    "S2 temperature given in eV, not K":       workflow(temperature_arg=300 * units.kB),
    "S3 forces in kcal/mol, energy in eV":     workflow("force_kcal"),
    "S4 force sign slip":                      workflow("force_sign"),
    "S5 per atom energy reported as total":    workflow("per_atom_energy"),
    "S6 leftover external field term":         workflow("field_term"),
    "S7 periodic boundaries switched off":     workflow(pbc=False),
    "S8 non conservative (direct force) model": workflow("nonconservative"),
    "S9 noisy forces (unconverged SCF)":       workflow("noisy_forces"),
    "S10 time step too large (8 fs)":          workflow(dt=8 * units.fs),
}


def naive_agent_tests(wf):
    """Typical self tests an agent writes: runs, finite, negative energy,
    zero forces on the perfect crystal, lattice constant near the expected value."""
    a = wf["make_atoms"]()
    a.calc = wf["calc"]
    e = a.get_potential_energy()
    f = a.get_forces()
    ok = np.isfinite(e) and np.all(np.isfinite(f))
    ok &= np.abs(f).max() < 1e-3
    vols, ens = [], []
    for s in np.linspace(0.97, 1.03, 7):
        b = wf["make_atoms"]()
        b.set_cell(b.cell * s, scale_atoms=True)
        b.calc = wf["calc"]
        vols.append(b.get_volume())
        ens.append(b.get_potential_energy())
    try:
        v0, _, _ = EquationOfState(vols, ens).fit()
        a_fit = (v0 / 27) ** (1 / 3)
        ok &= abs(a_fit - A0) / A0 < 0.05
    except Exception:
        ok = False
    return bool(ok)


if __name__ == "__main__":
    out = {}
    print(f"{'case':44s} naive  tripwires fired")
    for name, wf in CASES.items():
        naive_pass = naive_agent_tests(wf)
        res = tw.run_all(wf)
        fired = [k for k, (p, _) in res.items() if not p]
        out[name] = {"naive_pass": naive_pass,
                     "tripwires": {k: [bool(p), float(v)] for k, (p, v) in res.items()}}
        print(f"{name:44s} {'pass' if naive_pass else 'FAIL':5s}  {', '.join(fired) or '(none)'}")
    faulty = [k for k in CASES if not k.startswith("reference")]
    naive_caught = sum(not out[k]["naive_pass"] for k in faulty)
    tw_caught = sum(any(not v[0] for v in out[k]["tripwires"].values()) for k in faulty)
    ref_clean = all(v[0] for v in out["reference (correct)"]["tripwires"].values())
    print(f"\nSilent errors caught: naive agent style tests {naive_caught}/{len(faulty)}, "
          f"Physics Tripwires {tw_caught}/{len(faulty)}; reference passes all tripwires: {ref_clean}")
    json.dump(out, open("pilot_results.json", "w"), indent=1)

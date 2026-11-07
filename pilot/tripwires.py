"""Physics Tripwires pilot: engine agnostic, physics derived checks for atomistic workflows.

Each check takes the calculator and settings exactly as a (possibly AI written) workflow
configured them, and tests a relation that must hold regardless of who wrote the code.
"""
import numpy as np
from ase import units
from ase.build import bulk
from ase.md.verlet import VelocityVerlet
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary


def _e(atoms, calc):
    a = atoms.copy()
    a.calc = calc
    return a.get_potential_energy(), a.get_forces()


def translation_invariance(make_atoms, calc, tol=1e-6):
    a = make_atoms()
    a.rattle(0.05, seed=1)
    e0, _ = _e(a, calc)
    b = a.copy()
    b.translate([0.37, -1.21, 2.05])
    e1, _ = _e(b, calc)
    dev = abs(e1 - e0) / len(a)
    return dev < tol, dev


def rotation_invariance(make_atoms, calc, tol=1e-5):
    a = make_atoms()
    a.rattle(0.05, seed=2)
    e0, f0 = _e(a, calc)
    b = a.copy()
    b.rotate(37, "z", rotate_cell=True)
    b.rotate(21, "x", rotate_cell=True)
    e1, f1 = _e(b, calc)
    # rotate reference forces the same way
    c = a.copy()
    c.set_positions(f0)
    c.rotate(37, "z", rotate_cell=False, center=(0, 0, 0))
    c.rotate(21, "x", rotate_cell=False, center=(0, 0, 0))
    fdev = np.abs(c.get_positions() - f1).max()
    edev = abs(e1 - e0) / len(a)
    return (edev < tol and fdev < 10 * tol), max(edev, fdev)


def permutation_invariance(make_atoms, calc, tol=1e-6):
    a = make_atoms()
    a.rattle(0.05, seed=3)
    e0, f0 = _e(a, calc)
    p = np.random.default_rng(0).permutation(len(a))
    e1, f1 = _e(a[p], calc)
    dev = max(abs(e1 - e0) / len(a), np.abs(f0[p] - f1).max())
    return dev < tol, dev


def force_energy_consistency(make_atoms, calc, h=1e-4, tol=2e-3):
    """Forces must equal minus the gradient of the energy (catches sign, unit, noise errors)."""
    a = make_atoms()
    a.rattle(0.08, seed=4)
    _, f = _e(a, calc)
    rng = np.random.default_rng(5)
    worst = 0.0
    for _ in range(6):
        i, k = rng.integers(len(a)), rng.integers(3)
        ap, am = a.copy(), a.copy()
        ap.positions[i, k] += h
        am.positions[i, k] -= h
        fd = -(_e(ap, calc)[0] - _e(am, calc)[0]) / (2 * h)
        worst = max(worst, abs(fd - f[i, k]) / max(1.0, abs(fd)))
    return worst < tol, worst


def closed_loop_work(make_atoms, calc, tol=1e-4, n=60):
    """Work of the forces around a closed path must vanish for conservative forces."""
    a = make_atoms()
    a.rattle(0.05, seed=6)
    a.calc = calc
    i = 0
    r0 = a.positions[i].copy()
    t = np.linspace(0, 2 * np.pi, n + 1)
    path = r0 + 0.15 * np.stack([np.cos(t) - 1, np.sin(t), 0 * t], axis=1)
    w = 0.0
    for j in range(n):
        mid = 0.5 * (path[j] + path[j + 1])
        a.positions[i] = mid
        w += a.get_forces()[i] @ (path[j + 1] - path[j])
    a.positions[i] = r0
    return abs(w) < tol, abs(w)


def extensivity(element, a0, calc, pbc=True, tol=2e-4):
    """Energy per atom must not depend on supercell size at fixed density."""
    small = bulk(element, "fcc", a=a0, cubic=True) * (2, 2, 2)
    large = bulk(element, "fcc", a=a0, cubic=True) * (3, 3, 3)
    small.pbc = large.pbc = pbc
    es = _e(small, calc)[0] / len(small)
    el = _e(large, calc)[0] / len(large)
    dev = abs(es - el)
    return dev < tol, dev


def equipartition(make_atoms, calc, temperature_arg, target_K, rel_tol=0.35):
    """Kinetic temperature after velocity initialisation must match the requested target."""
    a = make_atoms()
    a.calc = calc
    MaxwellBoltzmannDistribution(a, temperature_K=temperature_arg, rng=np.random.default_rng(7))
    Stationary(a)
    t = a.get_temperature()
    dev = abs(t - target_K) / target_K
    return dev < rel_tol, dev


def nve_drift(make_atoms, calc, dt, temperature_arg, steps=200, tol=2e-3):
    """Total energy in NVE must be conserved within a small bound (catches bad time steps,
    unit slips and non conservative forces)."""
    a = make_atoms()
    a.calc = calc
    MaxwellBoltzmannDistribution(a, temperature_K=temperature_arg, rng=np.random.default_rng(8))
    Stationary(a)
    e0 = a.get_total_energy()
    dyn = VelocityVerlet(a, timestep=dt)
    worst = 0.0
    for _ in range(steps // 10):
        dyn.run(10)
        etot = a.get_total_energy()
        if not np.isfinite(etot):
            return False, float("inf")
        worst = max(worst, abs(etot - e0) / len(a))
    return worst < tol, worst


ALL = [
    "translation", "rotation", "permutation", "force_energy",
    "closed_loop", "extensivity", "equipartition", "nve_drift",
]


def run_all(wf):
    """wf: dict describing the workflow as configured (calc, make_atoms, dt, temperature_arg...)."""
    m, c = wf["make_atoms"], wf["calc"]
    res = {
        "translation": translation_invariance(m, c),
        "rotation": rotation_invariance(m, c),
        "permutation": permutation_invariance(m, c),
        "force_energy": force_energy_consistency(m, c),
        "closed_loop": closed_loop_work(m, c),
        "extensivity": extensivity(wf["element"], wf["a0"], c, pbc=wf["pbc"]),
        "equipartition": equipartition(m, c, wf["temperature_arg"], wf["target_K"]),
        "nve_drift": nve_drift(m, c, wf["dt"], wf["temperature_arg"]),
    }
    return res

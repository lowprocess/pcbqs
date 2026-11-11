"""
A general switching-circuit simulator on ngspice: the circuit is a model file and nothing in the code names a
converter.

    python -m pcbqs.circuit model.json --out results        # run it (every point of its sweep)
    python -m pcbqs.circuit --templates                     # the example circuits to start from
    python -m pcbqs.circuit --template boost > model.json

    from pcbqs.circuit import load, run
    summary = run(load('model.json'), 'results')

A model (model.py) is parts, couplings, drives, probes, measures and targets, with parameters its values may refer
to and a sweep over any of them:

    parts       every element, by type and nodes: resistor, capacitor, inductor, diode, switch, source, load
    couplings   pairs of inductors and their coupling factor (a transformer, a wireless link)
    drives      named gate signals: frequency, duty, phase, dead time; a switch names its drive, or its complement
    probes      named voltages between nodes and currents through parts
    measures    a statistic of a probe expression over the last period: avg, rms, pp, max, min
    targets     a knob (any number of the model), a measure and the value it must reach

The engine (engine.py) starts from ngspice's own DC solution with every switch off, runs whole periods until every
capacitor voltage and inductor current repeats, moves each target's knob by a secant step until its measure is
met, and reports what follows from each part's type without being asked: its power, a switch's and a diode's
current and voltage stress, a capacitor's ripple, an inductor's ripple, the sources' and loads' power, the
efficiency and the power balance. deck.py writes the ngspice deck (on power_spice.Circuit, which has the passives
with their parasitics, the switch models and the run).
"""
from .model import load, validate, resolve, cases, templates, template, ModelError      # noqa: F401
from .engine import run, run_case                                                        # noqa: F401

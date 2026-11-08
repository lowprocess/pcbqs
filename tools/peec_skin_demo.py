"""Generate and solve a two-coupon PCB demonstrating displacement current and skin loss.

Run: python tools/peec_skin_demo.py --out examples/peec-skin/results
Only pcbqs is required. The generated KiCad board is also independently importable.
"""
import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import time
import uuid

import numpy as np
from scipy.constants import epsilon_0, mu_0
from shapely.geometry import box, mapping, shape

from pcbqs.mqs import MQS


SIGMA = 5.8e7
COPPER_MM = .070
GAP_MM = .300
ER = 4.0
TAND = .002
PLATE_MM = 20.0
LENGTH_MM = 32.0
WIDTH_MM = 6.0
LOAD_OHM = 1000.0


def board():
    """Model coordinates are mm, y up; both coupons share a specified laminate."""
    cap = box(4, 4, 24, 24)
    strip = box(4, 32, 36, 38)
    copper = {
        'F.Cu': [dict(net='CAP_TOP', polygon=mapping(cap)), dict(net='LOOP', polygon=mapping(strip))],
        'B.Cu': [dict(net='CAP_BOTTOM', polygon=mapping(cap)), dict(net='LOOP', polygon=mapping(strip))],
    }
    vias = [dict(x=35.5, y=y, r=.3, z0=0., z1=GAP_MM, net='LOOP')
            for y in np.linspace(32.7, 37.3, 5)]
    terminals = {'cap_top': [('F.Cu', box(4, 4, 5, 5))],
                 'cap_bottom': [('B.Cu', box(4, 4, 5, 5))],
                 'loop_top': [('F.Cu', box(4, 32, 5, 38))],
                 'loop_bottom': [('B.Cu', box(4, 32, 5, 38))]}
    footprints = []
    for name, pads in terminals.items():
        layer, geom = pads[0]
        net = 'LOOP' if name.startswith('loop') else 'CAP_TOP' if name.endswith('top') else 'CAP_BOTTOM'
        footprints.append(dict(ref=name.upper(), value='test contact', props={}, lib='pcbqs:Contact',
            side='F' if layer == 'F.Cu' else 'B', x=geom.centroid.x, y=geom.centroid.y, rot=0,
            pads=[dict(name='1', net=net, layers=[layer], xy=list(geom.centroid.coords[0]), geom=mapping(geom))]))
    b = dict(format='kicad2ems-board', version=1, units='mm', source='peec_skin_demo.py',
        outline=mapping(box(0, 0, 40, 44)), copper=copper, vias=vias, ports=[], parts=[],
        prims=dict(footprints=footprints),
        stack=dict(copper=[dict(name='B.Cu', z=0., t=COPPER_MM), dict(name='F.Cu', z=GAP_MM, t=COPPER_MM)],
                   layers=[dict(name='core', z0=0., z1=GAP_MM, er=ER, tand=TAND, kind='diel')],
                   thickness=GAP_MM))
    return b, terminals


def write_kicad(b, path):
    """KiCad 8 board with filled zones, contacts and five plated shorting vias.

    kicad2ems places thin copper sheets at dielectric interfaces and ignores copper
    thickness when accumulating z, so the declared dielectric thickness is GAP_MM.
    Copper thickness is retained separately for the conductor-impedance model.
    """
    def uid():
        return str(uuid.uuid4())
    nets = {'CAP_TOP': 1, 'CAP_BOTTOM': 2, 'LOOP': 3}
    lines = ['(kicad_pcb (version 20240108) (generator "pcbqs")',
        f'(general (thickness {GAP_MM + 2*COPPER_MM}))',
        '(layers (0 "F.Cu" signal) (31 "B.Cu" signal) (37 "F.SilkS" user)',
        ' (36 "B.SilkS" user) (39 "F.Mask" user) (38 "B.Mask" user)',
        ' (44 "Edge.Cuts" user) (49 "F.Fab" user) (48 "B.Fab" user))',
        '(setup (stackup',
        f' (layer "F.Cu" (type "copper") (thickness {COPPER_MM}))',
        f' (layer "dielectric 1" (type "core") (thickness {GAP_MM}) (material "Specified er=4 laminate") (epsilon_r {ER}) (loss_tangent {TAND}))',
        f' (layer "B.Cu" (type "copper") (thickness {COPPER_MM}))',
        ' (copper_finish "None") (dielectric_constraints no)) (pad_to_mask_clearance 0))',
        '(net 0 "")', *[f'(net {i} "{n}")' for n, i in nets.items()]]
    for layer, records in b['copper'].items():
        for rec in records:
            points = ' '.join(f'(xy {x:g} {-y:g})' for x, y in shape(rec['polygon']).exterior.coords[:-1])
            lines.append(f'(zone (net {nets[rec["net"]]}) (net_name "{rec["net"]}") (layer "{layer}") (uuid "{uid()}")'
                ' (hatch edge 0.5) (connect_pads yes (clearance 0.2)) (min_thickness 0.1) (filled_areas_thickness no) (fill yes (thermal_gap 0.2) (thermal_bridge_width 0.2))'
                f' (polygon (pts {points})) (filled_polygon (layer "{layer}") (pts {points})))')
    for fp in b['prims']['footprints']:
        pad = fp['pads'][0]; layer = pad['layers'][0]; x, y = pad['xy']
        x0, y0, x1, y1 = shape(pad['geom']).bounds
        silk = 'F.SilkS' if layer == 'F.Cu' else 'B.SilkS'
        mask = 'F.Mask' if layer == 'F.Cu' else 'B.Mask'
        effects = '(effects (font (size 1 1) (thickness .15))'+(' (justify mirror)' if layer=='B.Cu' else '')+')'
        lines.append(f'(footprint "Contact" (layer "{layer}") (uuid "{uid()}") (at {x:g} {-y:g})'
            f' (property "Reference" "{fp["ref"]}" (at 6 -1) (layer "{silk}") {effects} (uuid "{uid()}"))'
            f' (property "Value" "Contact" (at 0 0) (layer "{silk}") (hide yes) {effects} (uuid "{uid()}"))'
            f' (pad "1" smd rect (at 0 0) (size {x1-x0:g} {y1-y0:g}) (layers "{layer}" "{mask}") (net {nets[pad["net"]]} "{pad["net"]}") (uuid "{uid()}")))')
    for via in b['vias']:
        lines.append(f'(via (at {via["x"]} {-via["y"]}) (size 0.9) (drill 0.6) (layers "F.Cu" "B.Cu") (net 3) (uuid "{uid()}"))')
    lines.append(f'(gr_rect (start 0 -44) (end 40 0) (stroke (width .1) (type default)) (fill none) (layer "Edge.Cuts") (uuid "{uid()}"))')
    lines.append(')')
    path.write_text('\n'.join(lines)+'\n')


def model(b, terms, coupon, dmax, backend, workers, capacitance, loss, faces=2, shared=None):
    names = ('cap_top', 'cap_bottom') if coupon == 'cap' else ('loop_top', 'loop_bottom')
    nets = {'CAP_TOP', 'CAP_BOTTOM'} if coupon == 'cap' else {'LOOP'}
    q = MQS(b, nets, terminals={n: terms[n] for n in names}, dmax=dmax,
            capacitance=capacitance, dielectric='stack', conductor_loss=loss,
            conductor_faces=faces, fill_backend=backend, fill_workers=workers,
            factor_cache_bytes=0, log=lambda *_: None)
    if shared is not None:
        for name in ('nodes', 'tris', 'node_of', 'z_of', 't_of'):
            if not np.array_equal(getattr(q, name), getattr(shared, name)):
                raise RuntimeError('Cannot reuse kernels between different meshes')
        q._copper = shared._copper
        q._potential = shared._potential
        q._sheet_terms = shared._sheet_terms
        q.fill_info = copy.deepcopy(shared.fill_info)
    q.source('port', *names, part_h=0)
    return q


def independent_references(f):
    """Parallel-plate C, bulk DC resistance and the high-frequency skin asymptote.

    These use SI geometry and Maxwell's closed-form limits, not pcbqs kernels.
    The strip expressions omit end spreading and via losses.
    """
    delta = np.sqrt(1/(np.pi*f*mu_0*SIGMA))
    return dict(parallel_plate_C_F=epsilon_0*ER*(PLATE_MM*.001)**2/(GAP_MM*.001),
                strip_Rdc_ohm=2*((LENGTH_MM-1)*.001)/(SIGMA*WIDTH_MM*.001*COPPER_MM*.001),
                skin_depth_m=delta,
                strip_Rac_two_face_asymptote_ohm=2*((LENGTH_MM-1)*.001)/(WIDTH_MM*.001*2*SIGMA*delta))


def run_resolution(b, terms, dmax, frequencies, backend, workers):
    result = {}; shared = None
    variants = [('mqs_dc', False, 'dc', 2), ('mqs_skin2', False, 'slab', 2),
                ('peec_dc', True, 'dc', 2), ('peec_skin2', True, 'slab', 2),
                ('peec_skin1', True, 'slab', 1)]
    # Build slab first so its reusable sheet Gram matrices exist for every variant.
    variants = [variants[3], variants[1], variants[0], variants[2], variants[4]]
    for name, cap, loss, faces in variants:
        started = time.perf_counter()
        q = model(b, terms, 'loop', dmax, backend, workers, cap, loss, faces, shared)
        r = q.solve(frequencies)
        z = r.z['port']
        record = dict(Z_real_ohm=z.real.tolist(), Z_imag_ohm=z.imag.tolist(), triangles=len(q.tris),
                      fill=copy.deepcopy(q.fill_info), seconds=time.perf_counter()-started)
        # Unit peak current: average real input power equals 1/2 Re(Z).
        copper_power = []; continuity = []
        for i, f in enumerate(frequencies):
            current = r.i[i]
            power = .5*np.vdot(current, (q._mats['R']+q._resistance_delta(f))@current).real
            dielectric = np.pi*f*np.vdot(r.charge[i], r.potential[i]).imag if cap else 0.
            copper_power.append(float(power))
            if cap:
                residual = q._peec['A']@current+2j*np.pi*f*r.charge[i]
                # Source current is 1 A peak. Normalize to it, rather than the
                # tiny net charging current of this almost closed loop.
                continuity.append(float(np.linalg.norm(residual)))
            if abs(.5*z[i].real-power-dielectric) > 1e-6*max(.5*z[i].real, 1e-12):
                raise RuntimeError('Real input power does not balance copper and dielectric loss')
        record.update(copper_loss_W_at_1A_peak=copper_power, charge_continuity_residual_A=continuity)
        result[name] = record
        if shared is None:
            shared = q
        print(f'  {name}: {len(q.tris)} triangles, {record["seconds"]:.2f} s', flush=True)
    # Capacitance coupon: independently disconnected electrodes, no lumped capacitor.
    q = model(b, terms, 'cap', dmax, backend, workers, True, 'dc')
    r = q.solve([1e3, 1e6, 1e7, 30e6])
    C, _, labels = q.capacitance_matrix()
    drive = np.zeros(len(C)); drive[labels['cap_top']] = 1; drive[labels['cap_bottom']] = -1
    cfloat = 1/(drive@np.linalg.solve(C, drive))
    residual = np.array([np.linalg.norm(q._peec['A']@current+2j*np.pi*f*charge)
                         for f, current, charge in zip(r.f, r.i, r.charge)])
    result['capacitor'] = dict(frequencies_Hz=r.f.tolist(), Z_real_ohm=r.z['port'].real.tolist(),
        Z_imag_ohm=r.z['port'].imag.tolist(), floating_C_real_F=float(cfloat.real),
        floating_C_imag_F=float(cfloat.imag), triangles=len(q.tris),
        charge_continuity_residual_A=residual.tolist(),
        load_transfer_mag=(abs(LOAD_OHM/(LOAD_OHM+r.z['port']))).tolist())
    try:
        mqs = model(b, terms, 'cap', dmax, backend, workers, False, 'dc')
        mqs.solve([1e6])
    except ValueError as exc:
        if 'no loop to drive' not in str(exc):
            raise
        result['capacitor']['mqs_without_capacitance'] = str(exc)
    else:
        raise RuntimeError('MQS unexpectedly allowed current between disconnected capacitor plates')
    return result


def figures(out, b, f, resolutions, reference):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fine = resolutions[-1]['models']
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    ax = axes[0]
    ax.add_patch(Rectangle((0, 0), 40, 44, fill=False, edgecolor='black'))
    for rec in b['copper']['F.Cu']:
        g = shape(rec['polygon']); ax.fill(*g.exterior.xy, color='#d99136', alpha=.8)
    for v in b['vias']:
        ax.add_patch(plt.Circle((v['x'], v['y']), .45, color='#243b53'))
    ax.text(14, 14, 'A: 20 × 20 mm plates\nDC open\nAC displacement current', ha='center', fontsize=8)
    ax.text(20, 35, 'B: broadside strip pair', ha='center', fontsize=9)
    ax.text(20, 29, 'Drive at left; five shorting vias at right', ha='center', fontsize=8)
    ax.set(xlim=(-1, 41), ylim=(-1, 45), aspect='equal', xlabel='mm', ylabel='mm', title='One PCB, two independent coupons')
    ax = axes[1]; c = fine['capacitor']; ff = np.array(c['frequencies_Hz'])
    zz = np.array(c['Z_real_ohm'])+1j*np.array(c['Z_imag_ohm'])
    ax.loglog(ff, abs(zz), 'o-', label='PEEC: finite capacitive impedance')
    ax.loglog(ff, 1/(2*np.pi*ff*reference['parallel_plate_C_F']), '--', label='ε₀ εr A / h reference (no fringe)')
    ax.text(.06, .1, 'MQS without C: open circuit at every f', transform=ax.transAxes, fontsize=9)
    ax.set(xlabel='Frequency (Hz)', ylabel='|Z| (Ω)', title='A: omitting capacitance loses the AC path')
    ax.grid(True, which='both', alpha=.25); ax.legend(fontsize=8)
    ax = axes[2]
    for key, label, style in [('peec_dc', 'PEEC + DC copper', '--'), ('mqs_skin2', 'MQS + 2-face skin', ':'),
                              ('peec_skin2', 'PEEC + 2-face skin', '-'), ('peec_skin1', 'PEEC + 1-face skin', '-.')]:
        ax.loglog(f, 1000*np.array(fine[key]['Z_real_ohm']), style, label=label)
    high = reference['skin_depth_m'] < COPPER_MM*.001/4
    ax.loglog(f[high], 1000*reference['strip_Rac_two_face_asymptote_ohm'][high], color='gray', alpha=.6, label='2-face high-f asymptote')
    ax.set(xlabel='Frequency (Hz)', ylabel='Port resistance (mΩ)', title='B: DC copper understates AC dissipation')
    ax.grid(True, which='both', alpha=.25); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out/'comparison.png', dpi=160); plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('examples/peec-skin/results'))
    parser.add_argument('--mesh', default='3,2,1.5', help='decreasing mesh spacings, mm; at least two resolutions')
    parser.add_argument('--backend', default='auto')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--generate-only', action='store_true')
    args = parser.parse_args(argv)
    meshes = [float(x) for x in args.mesh.split(',')]
    if len(meshes)<2 or any(not np.isfinite(x) or x<=0 for x in meshes) or any(a<=b for a,b in zip(meshes, meshes[1:])):
        parser.error('--mesh needs at least two decreasing positive spacings')
    out = args.out; out.mkdir(parents=True, exist_ok=True)
    b, terms = board()
    (out/'board.json').write_text(json.dumps(b, indent=2)+'\n')
    write_kicad(b, out/'peec-skin.kicad_pcb')
    if args.generate_only:
        return
    f = np.geomspace(1e3, 30e6, 17)
    reference = independent_references(f)
    resolutions = []
    for dmax in meshes:
        print(f'Mesh spacing {dmax:g} mm', flush=True)
        models = run_resolution(b, terms, dmax, f, args.backend, args.workers)
        resolutions.append(dict(dmax_mm=dmax, models=models))
    previous, fine = [r['models'] for r in resolutions[-2:]]
    ratio = fine['peec_skin2']['Z_real_ohm'][-1]/fine['peec_dc']['Z_real_ohm'][-1]
    cap_change = abs(fine['capacitor']['floating_C_real_F']/previous['capacitor']['floating_C_real_F']-1)
    resistance_change = abs(fine['peec_skin2']['Z_real_ohm'][-1]/previous['peec_skin2']['Z_real_ohm'][-1]-1)
    reference_c = reference['parallel_plate_C_F']; c = fine['capacitor']['floating_C_real_F']
    checks = dict(capacitance_mesh_change=cap_change, skin_resistance_mesh_change=resistance_change,
                  skin_vs_dc_resistance_ratio_at_30MHz=ratio,
                  capacitance_relative_to_parallel_plate=c/reference_c,
                  dc_strip_reference_relative_error=abs(fine['mqs_dc']['Z_real_ohm'][0]/reference['strip_Rdc_ohm']-1),
                  skin_asymptote_relative_error_at_30MHz=abs(fine['peec_skin2']['Z_real_ohm'][-1]/reference['strip_Rac_two_face_asymptote_ohm'][-1]-1),
                  capacitance_static_vs_1kHz_relative_error=abs(-1/(2*np.pi*1e3*fine['capacitor']['Z_imag_ohm'][0])/c-1),
                  max_charge_continuity_residual_A=max(max(v['charge_continuity_residual_A'], default=0) for v in fine.values()),
                  PEEC_vs_MQS_skin_impedance_relative=float(np.max(abs(
                      (np.array(fine['peec_skin2']['Z_real_ohm'])+1j*np.array(fine['peec_skin2']['Z_imag_ohm'])) /
                      (np.array(fine['mqs_skin2']['Z_real_ohm'])+1j*np.array(fine['mqs_skin2']['Z_imag_ohm']))-1))))
    # These are physical discriminators plus a mesh sensitivity check, not implementation-mirroring tests.
    checks['passed'] = bool(cap_change<.05 and resistance_change<.05 and ratio>2 and .9<c/reference_c<1.4
        and checks['dc_strip_reference_relative_error']<.15
        and checks['skin_asymptote_relative_error_at_30MHz']<.15
        and checks['capacitance_static_vs_1kHz_relative_error']<.001
        and checks['max_charge_continuity_residual_A']<1e-9
        and checks['PEEC_vs_MQS_skin_impedance_relative']<.01)
    from pcbqs import __version__
    report = dict(format=1, pcbqs_version=__version__, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        board_sha256=hashlib.sha256((out/'board.json').read_bytes()).hexdigest(),
        frequencies_Hz=f.tolist(), settings=dict(sigma_Sm=SIGMA, copper_mm=COPPER_MM,
        gap_mm=GAP_MM, er=ER, tand=TAND, load_ohm=LOAD_OHM, backend=args.backend, workers=args.workers),
        independent_reference={k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in reference.items()},
        resolutions=resolutions, checks=checks,
        limits=['Synthetic laminate parameters; no hardware measurement or full-wave validation.',
                'Skin model is a finite-thickness slab; one/two-face cases are sensitivities, not rigorous bounds.',
                'Via barrels retain DC resistance; fixtures, roughness and connector parasitics are excluded.',
                'Both copper layers are represented as thin sheets at dielectric interfaces.',
                'Two meshes establish the reported sensitivity, not a general discretization error bound.'])
    (out/'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    with (out/'impedance.csv').open('w', newline='') as stream:
        writer = csv.writer(stream); writer.writerow(['f_Hz', *[v+'_'+p for v in fine if v!='capacitor' for p in ('R_ohm','X_ohm')]])
        for i, freq in enumerate(f):
            writer.writerow([freq, *[fine[v][p][i] for v in fine if v!='capacitor' for p in ('Z_real_ohm','Z_imag_ohm')]])
    figures(out, b, f, resolutions, reference)
    print(json.dumps(checks, indent=2))
    if not checks['passed']:
        raise SystemExit('Demonstration checks failed; inspect summary.json and refine the mesh')


if __name__ == '__main__':
    main()

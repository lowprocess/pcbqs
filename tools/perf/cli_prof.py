import sys, cProfile, pstats, time, resource
which = sys.argv[1]
if which == 'mqs':
    from pcbqs.mqs_cli import main
    argv = ['/usr/share/kicad/demos/cm5_minima/CM5_MINIMA_3.kicad_pcb', '--converter', 'U702', '--vin', '+5V', '--harmonics', '1,2,3,5,7', '--heights', '1', '--out', 'out_mqs'] + sys.argv[2:]
else:
    from pcbqs.pdn.cli import main
    caps = [f'CA{k}=47e-6,0.02,1e-9' for k in range(1, 5)]
    argv = ['/home/bt/Desktop/wpthb/wpthb.kicad_pcb', '--rail', 'VIN', '--volts', '24', '--source', 'TP2', '--ground-source', 'TP1',
            '--load', 'Q1=0.67', '--load', 'Q2=0.67', '--through', 'L3', '--dcr', 'L3=0.03',
            *[x for c in caps for x in ('--cap', c)], '--dmax', '0.4', '--out', 'out_pdn'] + sys.argv[2:]
pr = cProfile.Profile(); t = time.time(); pr.enable(); main(argv); pr.disable()
print('TOTAL', round(time.time() - t, 1), 's peak MB', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024)
st = pstats.Stats(pr); st.sort_stats('tottime').print_stats(16); st.sort_stats('cumtime').print_stats(r'pcbqs', 22)

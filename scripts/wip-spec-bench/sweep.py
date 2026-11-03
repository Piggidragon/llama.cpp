# usage: sweep.py: tokens per second per task and draft max (median of the reps), restores and tokens per step from the traces
import json, os, statistics as st, collections as C
here = os.path.dirname(os.path.abspath(__file__))
rows = [json.loads(l) for l in open(os.path.join(here, 'results.jsonl'))]
tasks = ['code-new', 'code-cpp', 'edit-code']
print('n_max ' + ' '.join(f'{t:>18s}' for t in tasks) + '   restores  tok/step')
base = {}
for n in range(1, 8):
    tag = f'd{n}'
    rs = [r for r in rows if r['tag'] == tag]
    if not rs:
        continue
    cells = []
    for t in tasks:
        xs = [r['tps'] for r in rs if r['task'] == t]
        med = st.median(xs) if xs else float('nan')
        if n == 2:
            base[t] = med
        cells.append((t, med, min(xs) if xs else 0, max(xs) if xs else 0))
    restores = 0; steps = 0; tok = 0
    tr = os.path.join(here, f'trace-{tag}.txt')
    if os.path.exists(tr):
        for l in open(tr):
            f = l.split()
            if f[0] == 'S':
                k, nacc = int(f[3]), int(f[4])
                steps += 1; tok += 1 + nacc
                if k - nacc > 2:
                    restores += 1
    out = []
    for t, med, lo, hi in cells:
        rel = f'{100*(med/base[t]-1):+4.0f}%' if t in base else '     '
        out.append(f'{med:6.1f} [{lo:5.1f}-{hi:5.1f}]{rel if n != 2 else "     "}')
    print(f'{n:5d} ' + ' '.join(out) + f'   {restores:8d}  {tok/max(1,steps):.2f}')

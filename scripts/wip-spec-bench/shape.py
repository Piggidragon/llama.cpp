# usage: shape.py TRACE: verify step time by whether the shape of the previous step was the same
import sys, statistics as st, collections as C
prev = None; plain = False; g = C.defaultdict(list); seen_d0 = False
for ln in open(sys.argv[1]):
    f = ln.split()
    if f[0] == 'D' and f[2] == '0':
        if seen_d0: plain = True
        seen_d0 = True
    if f[0] == 'S':
        n = int(f[6]); t = float(f[7])
        key = (n, 'after plain' if plain else ('same n' if prev == n else 'other n'))
        if f[5] == f[6] and f[9] == '1': g[key].append(t)
        prev = n; plain = False; seen_d0 = False
for k in sorted(g):
    v = g[k]
    if len(v) >= 5: print(k, len(v), 'median', int(st.median(v)))

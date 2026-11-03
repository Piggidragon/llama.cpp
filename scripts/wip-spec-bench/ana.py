# usage: ana.py TRACE: acceptance by position and draft probability, step and draft times
import sys, collections as C, statistics as st
edges = [0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99]
def pbin(p):
    b = 0
    while b < len(edges) and p >= edges[b]: b += 1
    return b
hit = C.defaultdict(float); tr = C.defaultdict(float)
ver = C.defaultdict(list); drt = C.defaultdict(list); depth = C.defaultdict(list); catch = []
D = []; steps = []; ks = C.Counter()
for ln in open(sys.argv[1]):
    f = ln.split()
    if f[0] == 'D': D.append(float(f[3]))
    elif f[0] == 'T': depth[int(f[1])].append(int(f[3]))
    elif f[0] == 'P': catch.append(int(f[2]))
    elif f[0] == 'S':
        seq, impl, k, nacc, nb, sn, t, td, clean, eq = int(f[1]), int(f[2]), int(f[3]), int(f[4]), int(f[5]), int(f[6]), float(f[7]), int(f[8]), int(f[9]), int(f[10])
        ks[(impl, k)] += 1
        if impl == 3 and len(D) >= k:
            for j in range(min(k, nacc + 1)):
                b = pbin(D[j]); tr[(j, b)] += 1; hit[(j, b)] += j < nacc
        if clean and eq and nb == sn: ver[sn].append(t)
        drt[(impl, k)].append(td)
        steps.append((impl, k, nacc, t + td))
        D = []
print('acceptance by position x p bin (hits/trials)')
print('pos ' + ' '.join(f'{("<" + str(edges[0])) if b == 0 else (">=" + str(edges[b-1])):>12s}' for b in range(8)))
for j in range(4):
    print(f'{j:3d} ' + ' '.join(f'{(hit[(j,b)]/tr[(j,b)] if tr[(j,b)] else float("nan")):5.2f}/{int(tr[(j,b)]):<6d}' for b in range(8)))
tot = sum(tr.values())
print('p bin share at pos 0..3:', [' '.join(f'{tr[(j,b)]/max(1,sum(tr[(j,c)] for c in range(8))):.2f}' for b in range(8)) for j in range(2)])
print('verify time by tokens (median us, n):', {n: (int(st.median(v)), len(v)) for n, v in sorted(ver.items()) if len(v) > 5})
print('draft time by (impl,k) (median us):', {k: int(st.median(v)) for k, v in sorted(drt.items()) if len(v) > 5})
print('draft depth time (median us):', {k: int(st.median(v)) for k, v in sorted(depth.items())})
print('catch-up (median us):', int(st.median(catch)) if catch else None)
print('steps by (impl,k):', dict(ks.most_common(12)))

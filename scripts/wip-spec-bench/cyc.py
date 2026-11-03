# usage: cyc.py TRACE...: per trace, steps with a verification vs draft calls without one, tokens and time of the verified steps
import sys, collections as C
for path in sys.argv[1:]:
    nS = 0; tok = 0; t = 0.0; d0 = 0; dropped = 0; D = []; restores = 0; kk = C.Counter()
    for ln in open(path):
        f = ln.split()
        if f[0] == 'D':
            if int(f[2]) == 0:
                if D and not D[-1][1]: dropped += 1
                D.append([float(f[3]), False]); d0 += 1
        elif f[0] == 'S':
            impl, k, nacc, tt, td = int(f[2]), int(f[3]), int(f[4]), float(f[7]), int(f[8])
            nS += 1; tok += 1 + nacc; t += tt + td
            if D: D[-1][1] = True
            if k - nacc > 2: restores += 1; kk[k] += 1
    print(path.split('/')[-1], f'verified steps {nS}, model draft calls {d0}, calls without a verification {d0 - sum(1 for x in D if x[1])}, tokens {tok}, ms/token {t/tok/1000:.2f}, restores (k-nacc>2) {restores} {dict(kk)}')

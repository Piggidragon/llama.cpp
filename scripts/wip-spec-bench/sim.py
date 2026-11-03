# usage: sim.py TRACE: tokens per time of draft-length policies on the MTP steps of a depth-4 trace, per request
import sys, collections as C
tasks = ['compose-de', 'chat-de', 'explain-en', 'code-new', 'edit-code'] * 2
V = {1: 27.8, 2: 29.3, 3: 30.6, 4: 32.4, 5: 34.3}
TD0, TD, CATCH = 2.0, 1.4, 0.8
edges = [0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99]
def pbin(p):
    b = 0
    while b < len(edges) and p >= edges[b]: b += 1
    return b
reqs = [[]]; D = []; last = None
for ln in open(sys.argv[1]):
    f = ln.split()
    if f[0] == 'P' and int(f[1]) > 8 and last == 'S': reqs.append([])
    if f[0] == 'D': D.append(float(f[3]))
    if f[0] == 'S':
        if int(f[2]) == 3 and int(f[3]) == 4 and len(D) >= 4: reqs[-1].append((D[:4], int(f[4])))
        D = []
    last = f[0]
reqs = [r for r in reqs if len(r) > 20]
def tdraft(k): return 0.0 if k == 0 else TD0 + TD*(k - 1)
def step(k, nacc, rows, pen):
    tok = 1 + min(k, nacc)
    t = tdraft(k) + V[k + 1] + CATCH
    if k > rows and nacc < k - rows: t += pen
    return tok, t
def run(policy, rows=4, pen=20.0):
    out = []
    for r in reqs:
        cal_h = C.defaultdict(lambda: 1.0); cal_t = C.defaultdict(lambda: 2.0)
        tok = tt = 0.0
        for ps, nacc in r:
            k = policy(ps, cal_h, cal_t, rows, pen)
            a, b = step(k, nacc, rows, pen); tok += a; tt += b
            for j in range(min(4, nacc + 1)):   # learn as the depth-4 trace shows it
                key = (j, pbin(ps[j])); cal_h[key] = cal_h[key]*0.98 + (j < nacc); cal_t[key] = cal_t[key]*0.98 + 1
        out.append(1000*tok/tt)
    return out
fixed = lambda k: (lambda ps, h, t, rows, pen: k)
def greedy(ps, h, t, rows, pen, cap=4):
    runs = []; E = 1.0; k = 0
    def rate(runs, k):
        e = 1 + sum(runs[:k]); tt = tdraft(k) + V[k + 1] + CATCH
        if k > rows: tt += (1 - runs[k - rows - 1]) * max(0, pen)
        return e / tt
    for i in range(cap):
        a = h[(i, pbin(ps[i]))] / t[(i, pbin(ps[i]))]
        r = (runs[-1] if runs else 1.0) * a
        if rate(runs + [r], i + 1) < rate(runs, i): break
        runs.append(r); k = i + 1
    return k
oracle = None
res = {}
for name, pol, rows in [('fix1', fixed(1), 4), ('fix2', fixed(2), 4), ('fix3', fixed(3), 4), ('fix4', fixed(4), 4), ('greedy rows4', greedy, 4), ('fix3 rows2', fixed(3), 2), ('fix4 rows2', fixed(4), 2), ('greedy rows2', greedy, 2), ('greedy rows1', greedy, 1), ('greedy rows0', greedy, 0), ('fix2 rows1', fixed(2), 1)]:
    res[name] = run(pol, rows)
base = res['fix2']
print('policy          ' + ' '.join(f'{t:>11s}' for t in tasks[:len(reqs)]))
for name, v in res.items():
    print(f'{name:15s} ' + ' '.join(f'{x:5.1f}({100*(x/b-1):+3.0f}%)' for x, b in zip(v, base)))

# usage: mkpreset.py BASE OUT key=value ... (replaces or adds keys in the [Qwen3.8-27b] section; value "-" removes)
import sys
base, out, *kvs = sys.argv[1:]
over = dict(kv.split('=', 1) for kv in kvs)
lines, sec, seen = [], None, set()
for ln in open(base).read().splitlines():
    if ln.startswith('['):
        if sec == '[Qwen3.8-27b]':
            lines += [f'{k} = {v}' for k, v in over.items() if k not in seen and v != '-']
        sec = ln.strip()
        lines.append(ln); continue
    k = ln.split('=', 1)[0].strip()
    if sec == '[Qwen3.8-27b]' and k in over:
        seen.add(k)
        if over[k] != '-':
            lines.append(f'{k} = {over[k]}')
        continue
    lines.append(ln)
if sec == '[Qwen3.8-27b]':
    lines += [f'{k} = {v}' for k, v in over.items() if k not in seen and v != '-']
open(out, 'w').write('\n'.join(lines) + '\n')

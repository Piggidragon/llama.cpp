# usage: bench.py PORT TAG [REPS] [TASKS] [PAR]; PAR=2 runs the tasks two at a time (two slots)
import json, sys, time, urllib.request, concurrent.futures as cf, os, statistics as st
port, tag = sys.argv[1], sys.argv[2]
reps = int(sys.argv[3]) if len(sys.argv) > 3 else 2
only = sys.argv[4].split(',') if len(sys.argv) > 4 and sys.argv[4] else None
par = int(sys.argv[5]) if len(sys.argv) > 5 else 1
here = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(here, 'edit-src.py')).read()
TASKS = {
    'compose-de': 'Schreibe einen ausführlichen, gut gegliederten Blogartikel (etwa 700 Wörter) auf Deutsch über eine Herbstwanderung in den Allgäuer Alpen: Vorbereitung, Route, Ausrüstung, Sicherheit und Tipps für Einsteiger.',
    'chat-de': 'Ich habe einen Raspberry Pi 5 und möchte darauf einen kleinen Heimserver mit Docker betreiben (Nextcloud, Pi-hole, Home Assistant). Erkläre mir Schritt für Schritt, wie ich vorgehe und worauf ich bei Speicher, Backups und Sicherheit achten muss.',
    'explain-en': 'Explain in detail how a B+ tree works: structure, search, insertion with node splits, deletion with merges, and why databases use it. Include a small worked example.',
    'code-new': 'Write a complete Python 3 script that parses an nginx access log, aggregates requests per status code, per path and per hour, detects IPs with more than 100 requests per minute, and prints a report. Use argparse, type hints and docstrings.',
    'code-cpp': 'Implement a thread-safe LRU cache in C++17 as a header-only class template LruCache<K, V> with get, put, erase and size, using std::list and std::unordered_map, and write a short main() with assertions that tests it.',
    'edit-code': 'Here is part of a Python file. Rename the class LlamaBenchData to BenchData everywhere, add a docstring to every method, and return the complete updated code in one code block.\n\n```python\n' + src + '```',
}
names = [n for n in TASKS if not only or n in only]
def run(name, rep):
    body = {'model': 'Qwen3.8-27b', 'messages': [{'role': 'user', 'content': TASKS[name]}], 'max_tokens': 1500, 'seed': 1000 + rep, 'cache_prompt': False}
    req = urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions', json.dumps(body).encode(), {'Content-Type': 'application/json'})
    t0 = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=900))
    t = r.get('timings', {})
    return {'tag': tag, 'task': name, 'rep': rep, 'tps': t.get('predicted_per_second'), 'n': t.get('predicted_n'), 'draft_n': t.get('draft_n'), 'draft_acc': t.get('draft_n_accepted'), 'wall': time.time() - t0, 'par': par}
res = []
jobs = [(n, r) for r in range(reps) for n in names]
with cf.ThreadPoolExecutor(par) as ex:
    if par == 1:
        for j in jobs:
            res.append(run(*j)); print(json.dumps(res[-1]), flush=True)
    else:
        for i in range(0, len(jobs), par):
            for x in ex.map(lambda j: run(*j), jobs[i:i + par]):
                res.append(x); print(json.dumps(x), flush=True)
with open(os.path.join(here, 'results.jsonl'), 'a') as f:
    for x in res:
        f.write(json.dumps(x) + '\n')
for n in names:
    xs = [x for x in res if x['task'] == n]
    acc = sum(x['draft_acc'] or 0 for x in xs) / max(1, sum(x['draft_n'] or 0 for x in xs))
    print(f'{tag:24s} {n:11s} tps med {st.median(x["tps"] for x in xs):6.1f}  accept {acc:.2f}  n {sum(x["n"] for x in xs)}')

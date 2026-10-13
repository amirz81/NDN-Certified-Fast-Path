from __future__ import annotations
import argparse, csv
from collections import Counter, defaultdict
from pathlib import Path

def depths(s): return [int(x) for x in s.split(',') if x.strip()]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--interests', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--depths', default='1,2,3,4')
    ap.add_argument('--min-count', type=int, default=20)
    ap.add_argument('--max-prefixes', type=int, default=1000)
    ap.add_argument('--max-per-depth', type=int, default=300)
    ap.add_argument('--action-depth', type=int, default=1)
    a=ap.parse_args()
    names=[]
    with Path(a.interests).open('r',encoding='utf-8',errors='ignore') as f:
        first=f.readline().strip()
        if first.startswith('/'): names.append(first)
        for line in f:
            s=line.strip()
            if s.startswith('/'): names.append(s)
    cnt=Counter()
    for name in names:
        comps=[c for c in name.split('/') if c]
        for d in depths(a.depths):
            if len(comps)>=d: cnt['/'+'/'.join(comps[:d])] += 1
    cand=[(p,c) for p,c in cnt.items() if c>=a.min_count and p!='/']
    by=defaultdict(list)
    for p,c in cand: by[len([x for x in p.split('/') if x])].append((p,c))
    kept=[]
    for d in sorted(by): kept.extend(sorted(by[d], key=lambda x:(-x[1],x[0]))[:a.max_per_depth])
    kept=sorted(set(kept), key=lambda x:(-x[1],x[0]))[:a.max_prefixes]
    kept=sorted(kept, key=lambda x:(len([c for c in x[0].split('/') if c]), x[0]))
    out=Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    rows=[]
    for p,c in kept:
        comps=[x for x in p.split('/') if x]
        action='/'.join(comps[:max(1,min(a.action_depth,len(comps)))]) if comps else 'root'
        rows.append((p,action,c,len(comps)))
    with out.open('w',encoding='utf-8',newline='') as f:
        wr=csv.writer(f); wr.writerow(['prefix','action','count','depth']); wr.writerows(rows)
    print('interests',len(names)); print('candidate_prefixes_after_min_count',len(cand)); print('written_prefixes',len(rows)); print('wrote',out); print('first 20:')
    for r in rows[:20]: print(r)
if __name__=='__main__': main()

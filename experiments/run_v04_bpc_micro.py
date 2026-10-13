from __future__ import annotations

import argparse, json, random
from pathlib import Path
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt

from ndnhw.name import NDNName
from ndnhw.trie import PrefixTrie
from ndnhw.trace import InterestTrace
from ndnhw.certificates import evaluate_certificate_aware_coverage
from ndnhw.bank_repair import repair_bank_greedy
from ndnhw.cert_bank_repair import cert_bank_repair
from ndnhw.banking import greedy_load_balance


def build_trace(n_groups=5, children_per_group=4, interests_per_child=8, seed=7):
    rng=random.Random(seed); trie=PrefixTrie(); action_ids={}; names=[]
    for g in range(n_groups):
        parent=f'/svc{g}/topic'; pid=trie.add_fib_prefix(parent, weight=0, access_load=0); action_ids[pid]=g
        for c in range(children_per_group):
            child=f'{parent}/sub{c}'; w=max(3, interests_per_child+rng.randint(-2,2))
            cid=trie.add_fib_prefix(child, weight=float(w), access_load=float(w))
            action_ids[cid]=g if c<children_per_group-1 else 1000+g
            for i in range(w): names.append(NDNName.parse(f'{child}/item{i}/seg'))
    counts=Counter()
    for name in names: counts[trie.lpm(name)]+=1
    for node in trie.nodes: node.fib_weight=node.access_load=0.0
    for node_id,count in counts.items(): trie.nodes[node_id].fib_weight=trie.nodes[node_id].access_load=float(count)
    return InterestTrace(names),trie,action_ids


def select_parents(trie,M,D):
    sel=set()
    for n in [n for n in trie.iter_nodes() if n.depth==2 and n.is_fib_prefix and n.depth<=D]:
        trial=sel|trie.closure({n.node_id}, include_root=False)
        if len(trial)<=M: sel=trial
    return sel

def select_children(trie,M,D):
    sel=set(); children=[n for n in trie.iter_nodes() if n.depth==3 and n.is_fib_prefix and n.depth<=D]
    children.sort(key=lambda n:n.fib_weight, reverse=True)
    for n in children:
        trial=sel|trie.closure({n.node_id}, include_root=False)
        if len(trial)<=M: sel=trial
    return sel

def add_row(rows, seed, regime, B, C, M, alg, sel, trace, trie, actions, cert_budget):
    _, br=greedy_load_balance(trie, sel, B=B, C=C)
    cr=evaluate_certificate_aware_coverage(trie, trace, sel, 8, action_ids=actions, max_certificates=cert_budget)
    rows.append({'seed':seed,'regime':regime,'B':B,'total_budget':M,'algorithm':alg,'prefix_nodes':len(sel),'cert_budget':cert_budget,'bank_feasible':br.feasible,'total_overload':br.total_overload,**cr.as_dict()})

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--outdir',default='outputs/v04_bpc_micro'); args=parser.parse_args()
    out=Path(args.outdir); out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for seed in [3,7]:
        trace,trie,actions=build_trace(seed=seed)
        total_load=sum(n.access_load for n in trie.nodes)
        for regime in ['fixed_total','fixed_per_bank']:
            B=2; C=total_load*0.35/B if regime=='fixed_total' else total_load*0.35
            for M in [16,24,32]:
                for cert_budget in [0,4,8]:
                    add_row(rows,seed,regime,B,C,M,'ChildExactGreedy+Cert',select_children(trie,M,8),trace,trie,actions,cert_budget)
                    add_row(rows,seed,regime,B,C,M,'ParentShortcut+Cert',select_parents(trie,max(0,M-cert_budget),8),trace,trie,actions,cert_budget)
                br=repair_bank_greedy(trie,M=M,D=8,B=B,C=C,max_rounds=3)
                add_row(rows,seed,regime,B,C,M,'BankRepairOnly',br.selected,trace,trie,actions,0)
                bpc=cert_bank_repair(trie,trace,total_budget=M,D=8,B=B,C=C,action_ids=actions,cert_budget_options=[0,4,8])
                # evaluate selected using its chosen cert budget for consistent report
                add_row(rows,seed,regime,B,C,M,'BPC-Repair',bpc.selected,trace,trie,actions,bpc.cert_budget)
    df=pd.DataFrame(rows); df.to_csv(out/'v04_bpc_micro_all.csv',index=False)
    summary=df.groupby(['regime','B','total_budget','algorithm'],as_index=False).agg(mean_certified_coverage=('certified_coverage','mean'),mean_certificate_gain=('certificate_gain','mean'),feasible_rate=('bank_feasible','mean'),mean_overload=('total_overload','mean'),mean_cert_budget=('cert_budget','mean'))
    summary.to_csv(out/'v04_bpc_micro_summary.csv',index=False)
    best=summary[summary.feasible_rate>0.999].sort_values('mean_certified_coverage',ascending=False)
    best.to_csv(out/'v04_bpc_micro_best_feasible.csv',index=False)
    plt.figure(); sub=summary[(summary.regime=='fixed_per_bank') & (summary.B==2)]
    for alg,g in sub.groupby('algorithm'):
        gg=g.sort_values('mean_certified_coverage').groupby('total_budget',as_index=False).tail(1)
        plt.plot(gg.total_budget,gg.mean_certified_coverage,marker='o',label=alg)
    plt.xlabel('Total budget'); plt.ylabel('Mean certified coverage'); plt.title('v0.4 micro: joint BPC vs baselines'); plt.legend(fontsize=7); plt.tight_layout(); plt.savefig(out/'plot_v04_micro_fixed_per_bank_B2.png',dpi=180); plt.close()
    compact={'best_feasible_rows':best.head(16).to_dict(orient='records'),'fixed_per_bank_B2':sub.to_dict(orient='records')}
    (out/'v04_bpc_micro_summary.json').write_text(json.dumps(compact,indent=2))
    print(json.dumps(compact,indent=2))
if __name__=='__main__': main()

from __future__ import annotations
import argparse, json, random, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
import matplotlib.pyplot as plt
from experiments.run_v043_trace_replay_validation import read_interests, read_fib, derive_fib, build_trace_trie, cert_options
from experiments.run_v04_bpc_lite import eval_row
from ndnhw.bank_repair import repair_bank_greedy
from ndnhw.cert_bank_repair import cert_bank_repair
from ndnhw.cert_bank_portfolio import cert_bank_portfolio

def ints(s): return [int(x) for x in s.split(',') if x.strip()]
def floats(s): return [float(x) for x in s.split(',') if x.strip()]
def sample_names(names,n,seed):
    if not n or n<=0 or len(names)<=n: return names
    rng=random.Random(seed); idx=sorted(rng.sample(range(len(names)),n)); return [names[i] for i in idx]

def run_cfg(trie,trace,action_ids,D,capacity_frac,cfg):
    regime,B,M,gamma,algorithms=cfg
    total_load=sum(n.access_load for n in trie.nodes)
    C=total_load*capacity_frac if regime=='fixed_per_bank' else total_load*capacity_frac/B
    rows=[]
    if 'BankRepairOnly' in algorithms:
        t=time.time(); bank=repair_bank_greedy(trie,M=M,D=D,B=B,C=C,max_rounds=3); br,cr=eval_row(trie,trace,bank.selected,D,B,C,action_ids,0)
        rows.append({'regime':regime,'B':B,'C':C,'total_budget':M,'gamma':gamma,'algorithm':'BankRepairOnly','source':'BankRepairOnly','runtime_s':time.time()-t,'prefix_nodes':len(bank.selected),'cert_budget':0,'bank_feasible':br.feasible,'total_overload':br.total_overload,**cr.as_dict()})
    opts=cert_options(M,gamma)
    if 'BPC-Repair' in algorithms:
        t=time.time(); bpc=cert_bank_repair(trie,trace,total_budget=M,D=D,B=B,C=C,action_ids=action_ids,cert_budget_options=opts,cert_cost=gamma); br,cr=eval_row(trie,trace,bpc.selected,D,B,C,action_ids,bpc.cert_budget)
        rows.append({'regime':regime,'B':B,'C':C,'total_budget':M,'gamma':gamma,'algorithm':'BPC-Repair','source':'BPC-Repair','runtime_s':time.time()-t,'prefix_nodes':len(bpc.selected),'cert_budget':bpc.cert_budget,'bank_feasible':br.feasible,'total_overload':br.total_overload,**cr.as_dict()})
    if 'BPC-Portfolio' in algorithms:
        t=time.time(); port=cert_bank_portfolio(trie,trace,total_budget=M,D=D,B=B,C=C,action_ids=action_ids,cert_budget_options=opts,cert_cost=gamma); br,cr=eval_row(trie,trace,port.selected,D,B,C,action_ids,port.cert_budget)
        rows.append({'regime':regime,'B':B,'C':C,'total_budget':M,'gamma':gamma,'algorithm':'BPC-Portfolio','source':port.source,'runtime_s':time.time()-t,'prefix_nodes':len(port.selected),'cert_budget':port.cert_budget,'bank_feasible':br.feasible,'total_overload':br.total_overload,**cr.as_dict()})
    return rows

def worker(wid,configs,args):
    start=time.time(); names=read_interests(Path(args['interests']),args.get('limit')); names=sample_names(names,args.get('max_interests_sample'),args.get('sample_seed',1))
    if args.get('fib'): prefixes,actions=read_fib(Path(args['fib']))
    else: prefixes,actions=derive_fib(names,args['derive_fib_depth'],args['min_fib_count'])
    trace,trie,action_ids=build_trace_trie(names,prefixes,actions)
    rows=[]
    for i,cfg in enumerate(configs):
        t=time.time(); rs=run_cfg(trie,trace,action_ids,args['D'],args['capacity_frac'],cfg)
        for r in rs: r['worker_id']=wid; r['config_index_in_worker']=i
        rows.extend(rs); print(f'[worker {wid}] config {i+1}/{len(configs)} {cfg[:4]} done in {time.time()-t:.2f}s', flush=True)
    meta={'worker_id':wid,'loaded_interests':len(names),'usable_interests_with_lpm':len(trace),'fib_prefixes':sum(1 for n in trie.nodes if n.is_fib_prefix),'trie_nodes':len(trie),'total_load':sum(n.access_load for n in trie.nodes),'configs':len(configs),'elapsed_s':time.time()-start}
    return wid,rows,meta

def plots(summary,outdir):
    for regime in sorted(summary.regime.unique()):
        for B in sorted(summary.B.unique()):
            sub=summary[(summary.regime==regime)&(summary.B==B)]
            if sub.empty: continue
            plt.figure()
            for alg,g in sub.groupby('algorithm'):
                g=g.sort_values('total_budget'); plt.plot(g.total_budget,g.mean_certified_coverage*100,marker='o',label=alg)
            plt.xlabel('Total budget'); plt.ylabel('Certified coverage (%)'); plt.title(f'Parallel trace replay: {regime}, B={B}'); plt.legend(fontsize=8); plt.tight_layout(); plt.savefig(outdir/f'plot_parallel_trace_{regime}_B{B}.png',dpi=220); plt.close()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--interests',required=True); ap.add_argument('--fib',default=None); ap.add_argument('--outdir',default='outputs/v044_real_trace_parallel'); ap.add_argument('--limit',type=int,default=None); ap.add_argument('--max-interests-sample',type=int,default=50000); ap.add_argument('--sample-seed',type=int,default=1); ap.add_argument('--derive-fib-depth',type=int,default=3); ap.add_argument('--min-fib-count',type=int,default=2); ap.add_argument('--D',type=int,default=8); ap.add_argument('--budgets',default='16,32,48'); ap.add_argument('--B-values',default='1,2,4'); ap.add_argument('--gammas',default='1.0'); ap.add_argument('--regimes',default='fixed_per_bank,fixed_total'); ap.add_argument('--capacity-frac',type=float,default=0.35); ap.add_argument('--workers',type=int,default=4); ap.add_argument('--algorithms',default='BankRepairOnly,BPC-Repair,BPC-Portfolio')
    a=ap.parse_args(); outdir=Path(a.outdir); outdir.mkdir(parents=True,exist_ok=True)
    algorithms=tuple(x.strip() for x in a.algorithms.split(',') if x.strip()); regimes=[x.strip() for x in a.regimes.split(',') if x.strip()]
    configs=[(regime,B,M,gamma,algorithms) for regime in regimes for B in ints(a.B_values) for M in ints(a.budgets) for gamma in floats(a.gammas)]
    workers=max(1,min(a.workers,len(configs))); chunks=[[] for _ in range(workers)]
    for i,c in enumerate(configs): chunks[i%workers].append(c)
    args={'interests':a.interests,'fib':a.fib,'limit':a.limit,'max_interests_sample':None if a.max_interests_sample==0 else a.max_interests_sample,'sample_seed':a.sample_seed,'derive_fib_depth':a.derive_fib_depth,'min_fib_count':a.min_fib_count,'D':a.D,'capacity_frac':a.capacity_frac}
    meta={'configs_total':len(configs),'workers':workers,'algorithms':list(algorithms),'regimes':regimes,'B_values':ints(a.B_values),'budgets':ints(a.budgets),'gammas':floats(a.gammas),'args':args}
    (outdir/'parallel_trace_metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    allrows=[]; metas=[]; start=time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs=[ex.submit(worker,wid,ch,args) for wid,ch in enumerate(chunks) if ch]
        for fut in as_completed(futs):
            wid,rows,wm=fut.result(); metas.append(wm); allrows.extend(rows); pd.DataFrame(rows).to_csv(outdir/f'parallel_chunk_worker_{wid}.csv',index=False); print(f'[main] worker {wid} rows={len(rows)} saved', flush=True)
    df=pd.DataFrame(allrows); df.to_csv(outdir/'parallel_trace_all.csv',index=False)
    summary=df.groupby(['regime','B','total_budget','gamma','algorithm'],as_index=False).agg(mean_certified_coverage=('certified_coverage','mean'),mean_certificate_gain=('certificate_gain','mean'),feasible_rate=('bank_feasible','mean'),mean_cert_budget=('cert_budget','mean'),mean_prefix_nodes=('prefix_nodes','mean'),mean_runtime_s=('runtime_s','mean'))
    summary.to_csv(outdir/'parallel_trace_summary.csv',index=False)
    src=df[df.algorithm=='BPC-Portfolio'].groupby('source',as_index=False).size().rename(columns={'size':'count'}); src.to_csv(outdir/'parallel_trace_source_totals.csv',index=False)
    plots(summary,outdir)
    compact={'elapsed_total_s':time.time()-start,'metadata':meta,'worker_metas':metas,'best_rows':summary.sort_values('mean_certified_coverage',ascending=False).head(15).to_dict(orient='records'),'portfolio_source_totals':src.to_dict(orient='records')}
    (outdir/'parallel_trace_summary.json').write_text(json.dumps(compact,indent=2),encoding='utf-8'); print(json.dumps(compact,indent=2))
if __name__=='__main__': main()

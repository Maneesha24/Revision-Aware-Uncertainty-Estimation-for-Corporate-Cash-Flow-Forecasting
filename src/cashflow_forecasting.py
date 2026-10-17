"""Point-in-time SEC vintage reconstruction and normalized conformal cash-flow intervals.

Dates have daily resolution:
reported facts become usable on the DAY AFTER their SEC 'filed' date. First-publication
cash flows are defined using the first usable YTD filing for the current period.
All aggregation and forecast features are filtered to the forecast origin.
SEC facts and accounting contexts still require independent validation.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
import time
from datetime import datetime, timezone
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.dummy import DummyRegressor

TAG = 'NetCashProvidedByUsedInOperatingActivities'
YTD_DURATIONS = {1: (60, 130), 2: (145, 224), 3: (225, 319), 4: (320, 415)}
BASE_FEATURES = ['q','lag1','lag2','lag3','lag4','lag5','lag6','lag7','lag8',
                 'mean4','std4','mean8','abs_change','log_scale','history_span']
REV_FEATURES = ['rev_count8','rev_magnitude8','rev_share8','rev_recency','revised_periods8']


def next_day(day: str) -> str:
    return (date.fromisoformat(day) + timedelta(days=1)).isoformat()


def get_json(url: str, cache_file: Path | None, user_agent: str, wait=0.16, retries=4) -> dict:
    if cache_file and cache_file.exists():
        return json.loads(cache_file.read_text())
    if not user_agent or '@' not in user_agent:
        raise ValueError('Set SEC_USER_AGENT to a real name/organization and contact email.')
    for i in range(retries):
        try:
            response = requests.get(url, headers={'User-Agent': user_agent, 'Accept-Encoding':'gzip, deflate'},timeout=45)
            if response.status_code in (429, 500, 502, 503, 504):
                time.sleep(min(12, 2 ** i)); continue
            response.raise_for_status()
            data = response.json()
            if cache_file:
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                cache_file.write_text(json.dumps(data))
            time.sleep(wait)
            return data
        except requests.RequestException:
            if i == retries - 1: raise
            time.sleep(2 ** i)
    raise RuntimeError(f'SEC request failed: {url}')


def quarter_from_duration(start: str, end: str, form: str) -> int | None:
    d = (date.fromisoformat(end) - date.fromisoformat(start)).days + 1
    options = [k for k,(low,high) in YTD_DURATIONS.items() if low <= d <= high]
    if len(options)!=1: return None
    q=options[0]
    if form.startswith('10-K') and q!=4: return None
    if form.startswith('10-Q') and q==4: return None
    return q


def extract_companyfacts(payload: dict, cik: int) -> list[dict]:
    """Extract YTD vintages, inferring fiscal quarter from duration for comparative facts."""
    series=payload.get('facts',{}).get('us-gaap',{}).get(TAG,{}).get('units',{}).get('USD',[])
    out={}
    for f in series:
        if not {'start','end','filed','accn','val','form'} <= f.keys(): continue
        if f['form'] not in {'10-Q','10-K','10-Q/A','10-K/A'}: continue
        d=(date.fromisoformat(f['end'])-date.fromisoformat(f['start'])).days+1
        q=next((q for q,(lo,hi) in YTD_DURATIONS.items() if lo<=d<=hi),None)
        if q is None or not isinstance(f['val'],(int,float)): continue
        item={'cik':cik,'fy_start':f['start'],'period_end':f['end'],'q':q,
             'accn':f['accn'],'filed':f['filed'],'public_date':next_day(f['filed']),
             'form':f['form'],'filing_fp':f.get('fp',''),'ytd':float(f['val'])}
        key=(cik,f['start'],f['end'],f['accn'])
        if key in out and out[key]['ytd']!=item['ytd']:out[key]['ambiguous']=True
        else:out[key]=item
    return [x for x in out.values() if not x.get('ambiguous')]


def select_initial_periods(facts: list[dict]) -> list[dict]:
    """Find 'current quarter' filing facts using accession/reportDate match from submissions.
    This avoids mistaking comparative FY duration rows for new quarterly disclosures.
    Requires facts augmented with 'is_current_filing' Boolean from SEC submissions.
    """
    grouped=defaultdict(list)
    for f in facts:
        if f.get('is_current_filing'):
            grouped[(f['fy_start'],f['q'],f['period_end'])].append(f)
    periods=[]
    for (fy,q,pe),rs in grouped.items():
        first=min(rs,key=lambda x:(x['public_date'],x['accn']))
        periods.append({'cik':first['cik'],'fy_start':fy,'q':q,'period_end':pe,
                        'first_public_date':first['public_date']})
    return sorted(periods,key=lambda x:(x['period_end'],x['q']))


def attach_current_filings(facts: list[dict], submissions: dict) -> list[dict]:
    """Match accession to period-ending report date from EDGAR submission metadata."""
    entries=[]
    for src in [submissions.get('filings',{}).get('recent',{})]:
        if src and 'accessionNumber' in src:
            entries.extend(zip(src['accessionNumber'],src.get('reportDate',[]),src.get('form',[])))
    exact={(accn,report):form for accn,report,form in entries if report}
    max_duration_q={}
    for f in facts:
        form=exact.get((f['accn'],f['period_end']))
        if form in ('10-Q','10-Q/A','10-K','10-K/A'):
            key=(f['accn'],f['period_end'])
            max_duration_q[key]=max(f['q'],max_duration_q.get(key,0))
    for f in facts:
        form=exact.get((f['accn'],f['period_end']))
        expected=4 if form and '10-K' in form else (
            int(f['filing_fp'][1]) if f.get('filing_fp','') in ('Q1','Q2','Q3') else
            max_duration_q.get((f['accn'],f['period_end']),0))
        f['is_current_filing']=bool(form in ('10-Q','10-Q/A','10-K','10-K/A')
            and expected==f['q'] and (('10-K' in form)==(f['q']==4)))
    return facts


def fetch_submissions_all(cik:int, cache_dir:Path, ua:str) -> dict:
    base=get_json(f'https://data.sec.gov/submissions/CIK{cik:010d}.json',
                  cache_dir/f'CIK{cik:010d}_submissions.json',ua)
    recent=base.get('filings',{}).get('recent',{})
    keys=('accessionNumber','reportDate','form')
    allrows=[dict(zip(keys,row)) for row in zip(*(recent.get(k,[]) for k in keys))]
    for f in base.get('filings',{}).get('files',[]):
        name=f.get('name')
        if not name:continue
        ext=get_json(f'https://data.sec.gov/submissions/{name}',cache_dir/name,ua)
        allrows.extend(dict(zip(keys,row)) for row in zip(*(ext.get(k,[]) for k in keys)))
    # Each quarterly/annual filing contributes an accession/reportDate current period.
    base['filings']['recent']={k:[r[k] for r in allrows] for k in keys}
    return base


def factual_versions(facts:list[dict]) -> dict[tuple,list[dict]]:
    out=defaultdict(list)
    for f in facts:out[(f['fy_start'],f['q'],f['period_end'])].append(f)
    for versions in out.values(): versions.sort(key=lambda f:(f['public_date'],f['accn']))
    return out


def vintage_ytd(versions:dict, p:dict, cutoff:str) -> float | None:
    candidates=[x for x in versions.get((p['fy_start'],p['q'],p['period_end']),[]) if x['public_date']<=cutoff]
    return candidates[-1]['ytd'] if candidates else None


def standalone_value(periods: list[dict],versions:dict,p:dict,cutoff:str) -> float | None:
    v=vintage_ytd(versions,p,cutoff)
    if v is None: return None
    if p['q']==1: return v
    prev=[x for x in periods if x['fy_start']==p['fy_start'] and x['q']==p['q']-1
          and x['period_end']<p['period_end']]
    if len(prev)!=1:return None
    older=vintage_ytd(versions,prev[0],cutoff)
    return None if older is None else v-older


def revision_metrics(periods:list[dict],versions:dict,cutoff:str,scale:float) -> dict:
    n,mag,revised,recent=0,0.0,0,0.0
    for p in periods:
        rs=[r for r in versions.get((p['fy_start'],p['q'],p['period_end']),[])
            if r['public_date']<=cutoff]
        if len(rs)<2: continue
        # Only count material changes between consecutively observed vintages.
        changes=[(rs[i]['ytd']-rs[i-1]['ytd'],rs[i]['public_date']) for i in range(1,len(rs))
                 if abs(rs[i]['ytd']-rs[i-1]['ytd'])>max(1.,abs(rs[i-1]['ytd'])*1e-9)]
        if not changes: continue
        revised+=1; n+=len(changes)
        mag+=sum(abs(d) for d,_ in changes)/max(scale,1.)
        last=max(ts for _,ts in changes)
        gap=(date.fromisoformat(cutoff)-date.fromisoformat(last)).days
        recent=max(recent,1/(1+max(gap,0)/365.25))
    return {'rev_count8':float(n),'rev_magnitude8':float(mag),
            'rev_share8':float(revised/max(len(periods),1)),
            'rev_recency':float(recent),'revised_periods8':float(revised)}


def make_panel(facts:list[dict]) -> tuple[pd.DataFrame,dict]:
    firm_groups=defaultdict(list)
    for f in facts:firm_groups[f['cik']].append(f)
    rows=[]; audit={'firms_requested':len(firm_groups),'firms_with_periods':0,'firms_with_rows':0,
                    'revisions_observed_in_rows':0,'rows_skipped_incomplete':0}
    for cik, ff in sorted(firm_groups.items()):
        periods=select_initial_periods(ff)
        if len(periods)>=6:audit['firms_with_periods']+=1
        versions=factual_versions(ff)
        before=len(rows)
        for j in range(8,len(periods)-1):
            p=periods[j]; target=periods[j+1]
            if (date.fromisoformat(target['period_end'])-date.fromisoformat(p['period_end'])).days>110:
                audit['rows_skipped_incomplete']+=1;continue
            cutoff=p['first_public_date']
            if target['first_public_date']<=cutoff: continue
            hist=periods[j-7:j+1]
            vals=[standalone_value(periods,versions,x,cutoff) for x in hist]
            target_value=standalone_value(periods,versions,target,target['first_public_date'])
            if any(x is None for x in vals) or target_value is None:
                audit['rows_skipped_incomplete']+=1;continue
            vals=np.array(vals,dtype=float)
            scale=max(1.,float(np.median(np.abs(vals))))
            nv=vals/scale
            f={'cik':cik,'origin':cutoff,'target_public_date':target['first_public_date'],
               'period_end':p['period_end'],'target_end':target['period_end'],
               'q':p['q'],'scale':scale,'y':float(target_value/scale),
               'y_usd':float(target_value),'last_usd':float(vals[-1])}
            for k in range(1,9):f[f'lag{k}']=float(nv[-k])
            f.update({'mean4':float(np.mean(nv[-4:])), 'std4':float(np.std(nv[-4:])),
                      'mean8':float(np.mean(nv)),'abs_change':float(abs(nv[-1]-nv[-2])),
                      'log_scale':float(np.log1p(scale)),
                      'history_span':float((date.fromisoformat(p['period_end'])-date.fromisoformat(hist[0]['period_end'])).days/365.25)})
            f.update(revision_metrics(hist,versions,cutoff,scale))
            if f['rev_count8']>0:audit['revisions_observed_in_rows']+=1
            rows.append(f)
        if len(rows)>before:audit['firms_with_rows']+=1
    return pd.DataFrame(rows), audit


def conformal_quantile(scores:np.ndarray,alpha:float)->float:
    scores=np.asarray(scores,dtype=float)
    scores=scores[np.isfinite(scores)]
    n=len(scores)
    if n<20:raise ValueError('At least 20 finite calibration scores required.')
    k=math.ceil((n+1)*(1-alpha))
    if k>n:return float('inf')
    return float(np.partition(scores,k-1)[k-1])


def model(min_samples_leaf: int = 20):
    """Build matched boosting models; the point model uses the default leaf size of 20."""
    if min_samples_leaf < 1:
        raise ValueError('min_samples_leaf must be >= 1')
    return HistGradientBoostingRegressor(max_iter=150, learning_rate=0.065,
        max_leaf_nodes=15,min_samples_leaf=min_samples_leaf,
        l2_regularization=1.,random_state=7)


def disclosure_safe_split(train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Block point-fitting labels that were unknown at any error-scale forecast origin.

    An origin-only split can include point-fitting labels published after scale
    fitting begins. Require every point-fitting label to be public strictly
    before the earliest scale-fitting origin to prevent that timing overlap.
    """
    dates = sorted(train.origin.unique())
    if len(dates) < 3:
        raise ValueError('Not enough historical dates to split point and scale fitting.')
    split = dates[max(0, int(len(dates)*.75)-1)]
    scale_train = train[train.origin > split].copy()
    if scale_train.empty:
        raise ValueError('No error-scale fitting rows after the origin split.')
    first_scale_origin = str(scale_train.origin.min())
    candidate_fit = train[train.origin <= split]
    fit = candidate_fit[candidate_fit.target_public_date < first_scale_origin].copy()
    if not fit.empty and not (fit.target_public_date.max() < first_scale_origin):
        raise AssertionError('Disclosure-date embargo failed.')
    audit = {
        'last_point_fit_origin': str(split),
        'first_scale_fit_origin': first_scale_origin,
        'last_point_fit_target_public_date': str(fit.target_public_date.max()) if len(fit) else None,
        'point_fit_labels_embargoed': int(len(candidate_fit) - len(fit)),
        'point_fit_rows_after_embargo': int(len(fit)),
        'scale_fit_rows': int(len(scale_train)),
        'point_labels_available_before_all_scale_origins': bool(
            len(fit) > 0 and fit.target_public_date.max() < first_scale_origin),
    }
    return fit, scale_train, audit


def score_intervals(y:np.ndarray, pred:np.ndarray, radius:np.ndarray, alpha:float)->dict:
    hit=(np.abs(y-pred)<=radius)
    loss=2*radius+(2./alpha)*np.maximum(np.abs(y-pred)-radius,0)
    return {'coverage':float(np.mean(hit)),'width_norm':float(np.mean(2*radius)),
            'interval_score_norm':float(np.mean(loss)), 'mae_norm':float(np.mean(np.abs(y-pred)))}


def experiment(df:pd.DataFrame, output:Path, train_end='2020-12-31',cal_end='2022-12-31',
               test_end='2025-12-31',alpha=0.1,scale_min_leaf=20):
    if not 0<alpha<1:raise ValueError('alpha must lie strictly between 0 and 1')
    output.mkdir(parents=True,exist_ok=True)
    df=df.sort_values(['origin','cik']).copy()
    train=df[(df.origin<=train_end)&(df.target_public_date<=train_end)]
    cal=df[(df.origin>train_end)&(df.origin<=cal_end)&(df.target_public_date<=cal_end)]
    test=df[(df.origin>cal_end)&(df.origin<=test_end)&(df.target_public_date<=test_end)]
    if min(len(train),len(cal),len(test))<40:
        raise ValueError(f'Not enough eligible observations: train={len(train)}, cal={len(cal)}, test={len(test)}; add more firms or choose feasible splits.')
    fit,scale_train,chronology_audit=disclosure_safe_split(train)
    (output/'chronology_audit.json').write_text(json.dumps(chronology_audit,indent=2)+'\n')
    if min(len(fit),len(scale_train))<30:raise ValueError('Insufficient rows for proper point/scale training.')
    center=model().fit(fit[BASE_FEATURES],fit.y)
    pred_fit_cal=center.predict(scale_train[BASE_FEATURES]); abs_err=np.abs(scale_train.y.to_numpy()-pred_fit_cal)
    err_target=np.log1p(abs_err)
    no_rev=model(scale_min_leaf).fit(scale_train[BASE_FEATURES],err_target)
    yes_rev=model(scale_min_leaf).fit(scale_train[BASE_FEATURES+REV_FEATURES],err_target)
    # Shuffle revision columns together to preserve their joint distribution
    # while breaking their alignment with scale-training outcomes.
    shuffled=scale_train[BASE_FEATURES+REV_FEATURES].copy()
    rng=np.random.default_rng(7); perm=rng.permutation(len(shuffled))
    for col in REV_FEATURES:shuffled[col]=shuffled[col].to_numpy()[perm]
    permuted=model(scale_min_leaf).fit(shuffled,err_target)
    pred_cal=center.predict(cal[BASE_FEATURES]); pred_test=center.predict(test[BASE_FEATURES]);
    methods={
       'constant':(np.ones(len(cal)),np.ones(len(test))),
       'feature_scale':(np.maximum(np.expm1(no_rev.predict(cal[BASE_FEATURES])),.02),
                        np.maximum(np.expm1(no_rev.predict(test[BASE_FEATURES])),.02)),
       'revision_scale':(np.maximum(np.expm1(yes_rev.predict(cal[BASE_FEATURES+REV_FEATURES])),.02),
                         np.maximum(np.expm1(yes_rev.predict(test[BASE_FEATURES+REV_FEATURES])),.02)),
       'permuted_revision':(np.maximum(np.expm1(permuted.predict(cal[BASE_FEATURES+REV_FEATURES])),.02),
                            np.maximum(np.expm1(permuted.predict(test[BASE_FEATURES+REV_FEATURES])),.02))}
    results=[];predictions=[]
    for name,(scale_c,scale_t) in methods.items():
        score=np.abs(cal.y.to_numpy()-pred_cal)/scale_c
        q=conformal_quantile(score,alpha)
        radius=q*scale_t
        overall=score_intervals(test.y.to_numpy(),pred_test,radius,alpha)
        results.append({'method':name,'subset':'all','n_train':len(train),'n_cal':len(cal),
                       'n_test':len(test),'q_cal':q,**overall})
        for subset,mask in [('history_revision',test.rev_count8.to_numpy()>0),
                            ('no_history_revision',test.rev_count8.to_numpy()==0)]:
            if mask.sum()>=10:
                results.append({'method':name,'subset':subset,'n_train':len(train),
                               'n_cal':len(cal),'n_test':int(mask.sum()),'q_cal':q,
                               **score_intervals(test.y.to_numpy()[mask],pred_test[mask],radius[mask],alpha)})
        for j,(_,r) in enumerate(test.iterrows()):
            predictions.append({'method':name,'cik':int(r.cik),'origin':r.origin,
              'target_end':r.target_end,'target_public_date':r.target_public_date,
              'revision_history':bool(r.rev_count8>0),'scale_usd':r.scale,
              'scale_hat_norm':float(scale_t[j]),
              'abs_error_norm':float(abs(r.y-pred_test[j])),
              'actual_usd':r.y_usd,'predicted_usd':pred_test[j]*r.scale,
              'lower_usd':(pred_test[j]-radius[j])*r.scale,
              'upper_usd':(pred_test[j]+radius[j])*r.scale,
              'covered':bool(abs(r.y-pred_test[j])<=radius[j]),
              'interval_score_norm':float(2*radius[j]+2/alpha*max(abs(r.y-pred_test[j])-radius[j],0.))})
    results_df=pd.DataFrame(results)
    results_df.to_csv(output/'metrics.csv',index=False)
    pd.DataFrame(predictions).to_csv(output/'predictions.csv',index=False)
    paired_bootstrap(pd.DataFrame(predictions),output)
    sample={k:int(v) for k,v in {'train':len(train),'point_fit':len(fit),'scale_fit':len(scale_train),
      'calibration':len(cal),'test':len(test),'test_revision_exposed':int((test.rev_count8>0).sum()),
      'distinct_firms_train':train.cik.nunique(),'distinct_firms_test':test.cik.nunique()}.items()}
    (output/'sample_counts.json').write_text(json.dumps(sample,indent=2))
    revision_diagnostic = {
        'scale_min_samples_leaf': int(scale_min_leaf),
        'scale_fit_revision_exposed': int((scale_train.rev_count8>0).sum()),
        'calibration_revision_exposed': int((cal.rev_count8>0).sum()),
        'test_revision_exposed': int((test.rev_count8>0).sum()),
        'scale_models_identical_test_predictions': bool(
            np.array_equal(methods['feature_scale'][1],methods['revision_scale'][1])),
        'source_of_hyperparameter': 'user-selected; pilot sensitivity only; do not optimize on the test set',
    }
    (output/'revision_diagnostic.json').write_text(json.dumps(revision_diagnostic,indent=2)+'\n')
    if revision_diagnostic['scale_fit_revision_exposed']<scale_min_leaf:
        print('WARNING: Revision-exposed scale-fitting rows fewer than min_samples_leaf; '
              'a direct split isolating revisions is impossible.')
    if revision_diagnostic['test_revision_exposed']<30:
        print('WARNING: Revision-exposed test subgroup too small for reliable subgroup claims.')
    if revision_diagnostic['scale_models_identical_test_predictions']:
        print('WARNING: Revision and feature-only scale models have identical test predictions.')

    make_latex(results_df,sample,output,alpha)
    return results_df,sample



def paired_bootstrap(pred:pd.DataFrame,output:Path,replicates:int=500)->None:
    """Paired firm-cluster percentile CI for proposed minus non-revision mean interval score.

    This conditions on trained models; it does not account for fit-stage uncertainty,
    time-wide shared macro shocks, or tuning. Interpret as descriptive sensitivity.
    """
    pivot=pred.pivot(index=['cik','origin','target_end'],columns='method',values='interval_score_norm')
    pair=pivot[['revision_scale','feature_scale']].dropna()
    delta=(pair.revision_scale-pair.feature_scale).rename('difference').reset_index()
    groups={key:values.difference.to_numpy() for key,values in delta.groupby('cik')}
    ciks=list(groups)
    rng=np.random.default_rng(7)
    draws=[]
    for _ in range(replicates):
        pick=rng.choice(ciks,size=len(ciks),replace=True)
        draws.append(float(np.mean(np.concatenate([groups[cik] for cik in pick]))))
    report={'n_firm_clusters':len(ciks),'point_delta':float(delta.difference.mean()),
            'ci_low':float(np.quantile(draws,.025)),
            'ci_high':float(np.quantile(draws,.975)),
            'ci_method':'Paired firm-cluster percentile bootstrap, 500 replicates, fixed models; descriptive'}
    (output/'paired_bootstrap.json').write_text(json.dumps(report,indent=2))


def make_latex(metrics:pd.DataFrame, counts:dict, output:Path, alpha:float):
    r=metrics[metrics.subset=='all'].set_index('method')
    names={'constant':'Constant conformal','feature_scale':'Non-revision scale',
           'revision_scale':'Revision-aware (proposed)','permuted_revision':'Shuffled-revision control'}
    lines=[r'\begin{table}[t]',r'\caption{Held-out test results (normalized units). All entries computed by supplied code.}',
      r'\label{tab:results}',r'\centering',r'\scriptsize',r'\begin{tabular}{lrrr}',r'\toprule',
      r'Method & Coverage & Width & Interval score\\',r'\midrule']
    for key,label in names.items():
        if key in r.index:
            x=r.loc[key];lines.append(f"{label} & {x.coverage:.3f} & {x.width_norm:.3f} & {x.interval_score_norm:.3f}\\\\")
    lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
    (output/'results_table.tex').write_text('\n'.join(lines)+'\n')
    (output/'sample_counts.tex').write_text(
      (r'\newcommand{\nTrain}{'+str(counts['train'])+'}\n'+
       r'\newcommand{\nCal}{'+str(counts['calibration'])+'}\n'+
       r'\newcommand{\nTest}{'+str(counts['test'])+'}\n'+
       r'\newcommand{\nFirms}{'+str(counts['distinct_firms_test'])+'}\n').replace('}\\n', '}\n'))


def synthetic_demo(nfirms=60, start='2015-01-01', end='2025-12-31')->pd.DataFrame:
    """Artificial demo to exercise code only; never evidence about SEC performance."""
    rng=np.random.default_rng(7); rows=[]
    dates=pd.date_range(start,end,freq='QE')
    for cik in range(1,nfirms+1):
        prior=rng.normal(0,1)
        for j,origin in enumerate(dates[:-1]):
            v=rng.normal(0,1)
            is_rev=int(rng.random()<.25)
            rmag=float(is_rev*rng.uniform(.01,.5))
            y=.55*prior+.15*v+rng.normal(0,.25+.5*rmag)
            f={'cik':cik,'origin':origin.date().isoformat(),
               'target_public_date':(dates[j+1]+pd.Timedelta(days=40)).date().isoformat(),
               'period_end':origin.date().isoformat(),'target_end':dates[j+1].date().isoformat(),
               'q':j%4+1,'scale':1e9,'y':y,'y_usd':y*1e9,'last_usd':prior*1e9,
               'rev_count8':float(is_rev),'rev_magnitude8':rmag,'rev_share8':float(is_rev)/8,
               'rev_recency':float(is_rev),'revised_periods8':float(is_rev),
               'mean4':prior/2,'std4':abs(v),'mean8':prior/2,'abs_change':abs(v),
               'log_scale':np.log1p(1e9),'history_span':1.75}
            for k in range(1,9):f[f'lag{k}']=prior+0.01*k
            rows.append(f);prior=y
    return pd.DataFrame(rows)


def main():
    p=argparse.ArgumentParser(description='Point-in-time SEC cash-flow uncertainty experiment')
    p.add_argument('--demo',action='store_true',help='Offline synthetic smoke test; not empirical SEC results')
    p.add_argument('--input-panel',help='Use a precomputed panel.csv; tagged external/unverified')
    p.add_argument('--ciks-file',default='configs/sample_ciks.txt')
    p.add_argument('--cache',default='cache')
    p.add_argument('--output',default='outputs')
    p.add_argument('--train-end',default='2020-12-31')
    p.add_argument('--cal-end',default='2022-12-31')
    p.add_argument('--test-end',default='2025-12-31')
    p.add_argument('--alpha',type=float,default=.1)
    p.add_argument('--scale-min-leaf',type=int,default=20,
                   help='Error-scale model min leaf; exploratory sensitivity ONLY on current pilot test')
    args=p.parse_args()
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    if args.demo and args.input_panel: p.error('--demo and --input-panel are mutually exclusive')
    if args.demo:
        source_type='synthetic'
        panel=synthetic_demo()
    elif args.input_panel:
        source_type='external_unverified'
        panel=pd.read_csv(args.input_panel,dtype={'origin':str,'target_public_date':str})
        missing=set(BASE_FEATURES+REV_FEATURES+['cik','origin','target_public_date',
                'target_end','scale','y','y_usd'])-set(panel.columns)
        if missing:raise ValueError(f'Input panel missing fields: {sorted(missing)}')
    else:
        source_type='sec'
        ua=os.environ.get('SEC_USER_AGENT','')
        if '@' not in ua: p.error('Set SEC_USER_AGENT="Full Name email@institution.edu"')
        ciks=[int(x.strip()) for x in Path(args.ciks_file).read_text().splitlines()
              if x.strip() and not x.lstrip().startswith('#')]
        allfacts=[];errors=[]
        for i,cik in enumerate(ciks,1):
            try:
                facts=get_json(f'https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json',
                  Path(args.cache)/f'CIK{cik:010d}_companyfacts.json',ua)
                subs=fetch_submissions_all(cik,Path(args.cache),ua)
                ff=extract_companyfacts(facts,cik)
                allfacts.extend(attach_current_filings(ff,subs))
                print(f'[{i}/{len(ciks)}] CIK {cik}: {len(ff)} YTD vintages')
            except Exception as e:
                print(f'WARN CIK {cik}: {e}')
                errors.append({'cik':cik,'error':str(e)})
        (output/'download_errors.json').write_text(json.dumps(errors,indent=2))
        if not allfacts:raise RuntimeError('No SEC facts downloaded; check SEC access and User-Agent.')
        hashes=[]
        for path in sorted(Path(args.cache).glob('*.json')):
            hashes.append({'file':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
        pd.DataFrame(hashes).to_csv(output/'source_hashes.csv',index=False)
        pd.DataFrame(allfacts).to_csv(output/'raw_vintages.csv',index=False)
        panel,audit=make_panel(allfacts)
        (output/'data_audit.json').write_text(json.dumps(audit,indent=2))
        if panel.empty:raise RuntimeError('No usable firm-quarters; inspect raw_vintages and audit.')
    panel.to_csv(output/'panel.csv',index=False)
    metrics,counts=experiment(panel,output,args.train_end,args.cal_end,args.test_end,args.alpha,
                             scale_min_leaf=args.scale_min_leaf)
    manifest={
        'data_source':source_type,'generated_utc':datetime.now(timezone.utc).isoformat(),
        'alpha':args.alpha,'train_end':args.train_end,'cal_end':args.cal_end,
        'test_end':args.test_end,'seed':7,'scale_min_leaf':args.scale_min_leaf,'target_concept':'us-gaap:'+TAG,
        'status':'completed_pipeline_not_independently_audited',
        'panel_sha256':hashlib.sha256((output/'panel.csv').read_bytes()).hexdigest(),
        'num_panel_rows':len(panel),'num_test_rows':counts['test'],
        'note':('Artificial demonstration, not SEC data.' if source_type=='synthetic'
                else 'Point-in-time SEC API processing; check source audits before publication.'
                     if source_type=='sec' else 'Unverified externally provided panel.'),
    }
    (output/'run_manifest.json').write_text(json.dumps(manifest,indent=2))
    print('EMBARGO',json.loads((output/'chronology_audit.json').read_text()))
    print('SAMPLE',counts)
    print(metrics[metrics.subset=='all'].to_string(index=False))
    print('Outputs:',output.resolve())

if __name__=='__main__':main()

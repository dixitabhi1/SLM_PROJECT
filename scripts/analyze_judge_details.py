import glob
import json
import os
from collections import defaultdict

def analyze():
    files = glob.glob('logs/judge_keys/key_*.json')
    q_stats = defaultdict(lambda: {'success': 0, 'failed': 0, 'slm_wins': 0, 'base_wins': 0, 'ties': 0})
    baseline_stats = defaultdict(lambda: {'slm_wins': 0, 'base_wins': 0, 'ties': 0})
    system_scores = defaultdict(lambda: {'correctness': [], 'completeness': [], 'coherence': []})
    pos_wins = {'Candidate A': 0, 'Candidate B': 0, 'Tie': 0}
    order_pairs = defaultdict(dict)
    differentiators = defaultdict(int)
    sample_reasons = []

    for kf in files:
        with open(kf, 'r', encoding='utf-8') as fp:
            kd = json.load(fp)
        qid = kd['query_id']
        if kd.get('status') != 'SUCCESS':
            q_stats[qid]['failed'] += 1
            continue
        q_stats[qid]['success'] += 1
        winner = kd['unblinded_winner']
        cand_a_sys = kd['candidate_a_system']
        cand_b_sys = kd['candidate_b_system']
        alias = kd['selected_alias']
        pos_wins[alias] += 1
        
        # load public log
        pub_file = kd['public_log_file']
        diff = 'unknown'
        reason = ''
        if os.path.exists(pub_file):
            with open(pub_file, 'r', encoding='utf-8') as pfp:
                pd = json.load(pfp)
                diff = pd.get('primary_differentiator', 'unknown')
                reason = pd.get('reasoning', '')
        differentiators[diff] += 1
        if len(sample_reasons) < 8 and reason:
            sample_reasons.append((qid, kd['pair_key'], kd['order_tag'], alias, winner, diff, reason))

        # system scores
        for sys_name, sc in kd.get('scores_by_system', {}).items():
            for crit in ['correctness', 'completeness', 'coherence']:
                if crit in sc:
                    system_scores[sys_name][crit].append(sc[crit])

        # pairing
        pair_id = (qid, kd['pair_key'])
        order_pairs[pair_id][kd['order_tag']] = kd

        # SLM vs baseline stats
        if cand_a_sys == 'slm_pipeline_v2':
            base = cand_b_sys
            if winner == 'slm_pipeline_v2':
                baseline_stats[base]['slm_wins'] += 1
                q_stats[qid]['slm_wins'] += 1
            elif winner == base:
                baseline_stats[base]['base_wins'] += 1
                q_stats[qid]['base_wins'] += 1
            else:
                baseline_stats[base]['ties'] += 1
                q_stats[qid]['ties'] += 1
        elif cand_b_sys == 'slm_pipeline_v2':
            base = cand_a_sys
            if winner == 'slm_pipeline_v2':
                baseline_stats[base]['slm_wins'] += 1
                q_stats[qid]['slm_wins'] += 1
            elif winner == base:
                baseline_stats[base]['base_wins'] += 1
                q_stats[qid]['base_wins'] += 1
            else:
                baseline_stats[base]['ties'] += 1
                q_stats[qid]['ties'] += 1

    print('=== QUERIES ===')
    for qid in sorted(q_stats.keys()):
        s = q_stats[qid]['success']
        f = q_stats[qid]['failed']
        sw = q_stats[qid]['slm_wins']
        bw = q_stats[qid]['base_wins']
        print(f'{qid}: success={s}, failed={f}, slm_wins={sw}, base_wins={bw}')

    print('\n=== BASELINES VS SLM ===')
    for b in sorted(baseline_stats.keys()):
        s_w = baseline_stats[b]['slm_wins']
        b_w = baseline_stats[b]['base_wins']
        tot = s_w + b_w
        pct = (s_w / tot * 100) if tot else 0
        print(f'{b}: SLM={s_w}, Base={b_w}, Total={tot}, SLM Win%={pct:.1f}%')

    print('\n=== POSITION WINS ===')
    for p, count in pos_wins.items():
        print(f'{p}: {count} ({count/sum(pos_wins.values())*100:.1f}%)')

    print('\n=== PRIMARY DIFFERENTIATORS ===')
    for d, count in differentiators.items():
        print(f'{d}: {count}')

    print('\n=== SYSTEM SCORES (MEAN) ===')
    for s in sorted(system_scores.keys()):
        c_m = sum(system_scores[s]['correctness']) / len(system_scores[s]['correctness']) if system_scores[s]['correctness'] else 0
        comp_m = sum(system_scores[s]['completeness']) / len(system_scores[s]['completeness']) if system_scores[s]['completeness'] else 0
        coh_m = sum(system_scores[s]['coherence']) / len(system_scores[s]['coherence']) if system_scores[s]['coherence'] else 0
        n = len(system_scores[s]['correctness'])
        print(f'{s} (N={n}): Corr={c_m:.2f}, Comp={comp_m:.2f}, Coh={coh_m:.2f}')

    print('\n=== BIDIRECTIONAL AGREEMENT ===')
    agree = 0
    disagree = 0
    single = 0
    for pair_id, tags in order_pairs.items():
        if 'forward' in tags and 'swapped' in tags:
            w_fwd = tags['forward']['unblinded_winner']
            w_swp = tags['swapped']['unblinded_winner']
            if w_fwd == w_swp:
                agree += 1
            else:
                disagree += 1
        else:
            single += 1
    total_pairs = agree + disagree
    print(f'Full Pairs: {total_pairs}, Agree: {agree} ({agree/total_pairs*100:.2f}%), Disagree: {disagree}, Single: {single}')

    print('\n=== SAMPLE REASONING ===')
    for item in sample_reasons:
        print(f'Query: {item[0]} | Pair: {item[1]} | Order: {item[2]} | Alias: {item[3]} | Winner: {item[4]}')
        print(f'  Differentiator: {item[5]}')
        print(f'  Reasoning: {item[6]}')
        print('-' * 60)

if __name__ == '__main__':
    analyze()


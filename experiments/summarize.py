"""Generate the requested Markdown report only from on-disk experiment evidence."""
import json
import os
import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


QROOT=Path('Experiment_record/Qwen3.5-9B-Base-PCH/rebuttal_v2')
LROOT=Path('Experiment_record/Llama-3-8B-PCH/rebuttal_v2')
REPORT=Path('Experiment_record/supplementary_results.md')
LABELS={'FIT_filter_off':'FIT（完整）','NPO_KL_filter_off':'NPO+KL（无过滤）',
        'NPO_KL_filter_on':'NPO+KL（仅加过滤）','RL_filter_off':'RLabel（无过滤）',
        'RL_filter_on':'RLabel（仅加过滤）','ALKN_filter_off':'ALKN（论文复现）',
        'O3_filter_off':'O³（PCH 适配）','budget_FIT':'FIT（等预算）',
        'budget_NPO_KL':'NPO+KL（等预算）','budget_ALKN':'ALKN（等预算）','budget_O3':'O³（等预算）'}


def display_name(path):
    return LABELS.get(path.name,path.name)


def run_name(path):
    model='Qwen3.5-9B-Base' if path.is_relative_to(QROOT) else 'Llama-3-8B'
    return model+' / '+display_name(path)


def read(path,default=None):
    if not path.exists(): return default
    return json.loads(path.read_text())


def number(value):
    return '—' if value is None else f'{value:.4f}'


def percentile(samples,p):
    if not samples: return None
    position=(len(samples)-1)*p
    lower=int(position)
    upper=min(lower+1,len(samples)-1)
    return samples[lower]+(samples[upper]-samples[lower])*(position-lower)


def exact_answer(answer,response):
    gold=' '.join(re.findall(r'\w+',answer.lower()))
    text=' '.join(re.findall(r'\w+',response.lower()))
    return bool(gold) and (' '+gold+' ') in (' '+text+' ')


def corrected_behavior(curves,path,initial=False):
    bank=read(Path('Experiment_record/probe_bank.json'),{'records':[]})
    gold={row['index']:row['answer'] for row in bank['records']}
    for stage in curves:
        rows=read(path/('baseline_responses.json' if initial else f"responses_stage{stage['requests']}.json"))
        if rows is None: continue
        if initial: rows=rows['records']
        behavior=stage.get('behavioral',stage)
        boundary=stage['requests']
        for key,selected in [('forget',rows[:boundary]),('retain',rows[boundary:])]:
            if selected:
                behavior[key]['exact_answer_leakage_rate']=sum(
                    exact_answer(row['gold'] if 'gold' in row else gold[row['index']],row['answer'])
                    for row in selected)/len(selected)
    return curves


def probes(path):
    data=read(path)
    if data and 'details' in data:
        # Recompute from per-generation evidence, including verbose generations
        # containing the full answer even when their full-response F1 is low.
        gold={row['index']:row['answer'] for row in read(Path('Experiment_record/probe_bank.json'))['records']}
        for row in data['details']:
            row['exact_leak']=exact_answer(gold[row['index']],row['answer'])
        found={r['index'] for r in data['details'] if r['exact_leak'] or r['token_f1']>=.8}
        data['extracted_indices']=sorted(found)
        data['any_prompt_extraction_rate']=len(found)/max(1,data['n_targets'])
        data['by_variant']={}
        for kind in sorted({r['variant'] for r in data['details']}):
            items=[r for r in data['details'] if r['variant']==kind]
            data['by_variant'][kind]={'n':len(items),
                'exact_rate':sum(r['exact_leak'] for r in items)/len(items),
                'extraction_rate':sum(r['exact_leak'] or r['token_f1']>=.8 for r in items)/len(items)}
    return data


def run(path):
    config=read(path/'config.json',{})
    done=read(path/'completed.json')
    curves=read(path/'evaluations.json',[])
    if path.is_relative_to(LROOT):
        curves=corrected_behavior(curves,path)
    rows=[]
    log=path/'requests.jsonl'
    if log.exists():
        rows=[json.loads(line) for line in log.read_text().splitlines() if line.strip()]
    complete=bool(done and not done.get('pilot') and not config.get('pilot')
                  and [r['requests'] for r in curves]==[60,120,180,240,300]
                  and len(rows)==300)
    return {'path':path,'config':config,'done':done,'curves':curves,'requests':rows,
            'complete':complete,'status':'完成' if complete else '运行中/待核验' if config else '待运行'}


def main():
    lines=['# 补充实验结果汇总','',
           '更新时间：'+datetime.now(ZoneInfo('Asia/Hong_Kong')).strftime('%Y-%m-%d %H:%M:%S %Z'),'',
           '本文件由原始 JSON/JSONL 自动汇总。没有完成标记、未覆盖 300 次请求，或仅为 pilot 的运行，不计为完成。',
           '尚未产出的结果留空；本文件不会将论文数值、旧实验结果或估计值填作本轮实测结果。','']
    names=['FIT_filter_off','NPO_KL_filter_off','ALKN_filter_off','O3_filter_off',
           'NPO_KL_filter_on','RL_filter_off','RL_filter_on',
           'budget_FIT','budget_NPO_KL','budget_ALKN','budget_O3']
    runs=[run(QROOT/name) for name in names]+[run(LROOT/'CURaTE'),run(LROOT/'FIT')]
    completed=sum(r['complete'] for r in runs)
    preparation_log=Path('logs/qwen35_p0_queue.log')
    prep_text=preparation_log.read_text(errors='replace') if preparation_log.exists() else ''
    progress=re.findall(r'\[stage (\d+)\] sample=(\d+)/(\d+) updates=(\d+)',prep_text)
    lines += ['## 模型与数据准备','',
              '- Qwen 真实 9B 前向/反向/保存/重载/生成检查：'+('已通过' if '[smoke passed]' in prep_text else '待完成')+'。']
    if progress:
        stage,sample,total,updates=progress[-1]
        lines.append(f'- Qwen 当前 checkpoint 准备：stage {stage}，已处理 {sample}/{total} 条，累计 {updates} 次参数更新。')
    for stage in [0,300]:
        prepared=read(Path(f'Model/Qwen3.5-9B-Base-PCH/stage{stage}/completed.json'))
        lines.append(f"- Qwen stage {stage} checkpoint："+('已完成' if prepared else '未完成')+'。')
        if prepared:
            lines.append(f"  共享准备成本：{prepared['updates']} 次更新，训练 {prepared['training_seconds']:.2f} 秒，峰值显存 {prepared['peak_allocated_gib']:.3f} GiB。")
    benchmark=read(Path('Experiment_record/runtime_pilots/qwen_batch_benchmark.json'))
    if benchmark:
        lines.append(f"- 评估吞吐实测（同一组 64 条问题、最多 128 个新 token、两份完整 9B 模型驻留）：批量 16 用时 {benchmark['16']['seconds']:.2f} 秒，批量 64 用时 {benchmark['64']['seconds']:.2f} 秒；对应峰值显存 {benchmark['16']['peak_allocated_gib']:.2f}/{benchmark['64']['peak_allocated_gib']:.2f} GiB。两批次的长续写有 {benchmark['identical_answers']}/64 条全文一致，故正式参考值统一按批量 64 重算。此项是单次评估吞吐测量，不是在线遗忘加速比。")
    reference=read(QROOT/'reference.json')
    if reference:
        lines += ['', '| Qwen 共享参考端点 | 前 300 条 forget quality | 后 300 条 retain quality |',
                  '|---|---:|---:|']
        for key,label in [('initial','初始 checkpoint'),('retain','仅 retain 训练 checkpoint')]:
            lines.append(f"| {label} | {number(reference[key]['forget']['quality'])} | {number(reference[key]['retain']['quality'])} |")
        unchanged={}
        for subset in ['forget','retain']:
            a,b=reference['initial'][subset]['quality'],reference['retain'][subset]['quality']
            unchanged[subset]=max(0.,1-abs(a/b-1)) if b else 0.
        lines += ['',f"按相同定义，保持初始模型完全不变时，300 请求处的 FD/RU 分别为 {unchanged['forget']:.4f}/{unchanged['retain']:.4f}（由共享端点直接推导，非追加实跑或计时结果）。因此高 FD/RU 不能单独解释为已删除知识，需结合原始 quality、实际更新覆盖率及知识残留检查。"]
    lines.append('')
    lines += ['## 完成情况','',f'完整连续运行：**{completed}/{len(runs)}**。', '',
              '| 模型 | 实验 | 状态 | 已记录请求 | 已评估阶段 |','|---|---|---|---:|---|']
    for result in runs:
        stages=', '.join(str(x['requests']) for x in result['curves']) or '—'
        lines.append(f"| {'Qwen3.5-9B-Base' if result['path'].is_relative_to(QROOT) else 'Llama-3-8B'} | {display_name(result['path'])} | {result['status']} | {len(result['requests'])} | {stages} |")
    scheduler=read(Path('logs/all_experiments_state.json'),{})
    issues=scheduler.get('failures',[])+scheduler.get('external_queue_failures',[])
    if scheduler.get('resource_waits'):
        lines.append('')
    for pending in scheduler.get('resource_waits',[]):
        lines.append(f"- 硬件排队：{pending['task']} 等待 GPU {pending['gpu']}；当前空闲 {pending['free_mib']/1024:.2f} GiB，启动余量阈值为 {pending['startup_threshold_mib']/1024:.2f} GiB（调度阈值，非实测峰值）。")
    if issues:
        lines += ['', '### 运行错误/中断记录','']
        lines += ['- `'+json.dumps(item,ensure_ascii=False)+'`' for item in issues]
    lines += ['', '## 统一设置与口径','',
              '- PCH 共 600 条；前 300 条使用 seed 42 的同一排列逐次提出遗忘请求，余下未请求记录作为各阶段 retain 集。文本与 QA 保持对齐并记录哈希。',
              '- Qwen 使用官方 `Qwen/Qwen3.5-9B-Base` 的文本骨干，revision `68c46c4b3498877f3ef123c856ecfde50c39f404`；PCH 为纯文本任务，视觉塔不参与。Llama 使用项目既有 PCH 微调 checkpoint。',
              '- 保留任务表现按 PCH retain QA 的 ROUGE-L、token F1 与答复覆盖率报告。RL 即 RLabel；完整 FIT 始终启用过滤、层归因和重要性路由。',
              '- 本轮标记为 `rebuttal-v2`：只对生成答案计算 ROUGE，最多生成 128 个新 token；保留惩罚为 KL(reference || current)。不能与旧口径结果直接混表。',
              '- Qwen 的正式参考指标与所有方法统一采用生成批量 64；批量 16 的未完成首组与旧参考值保存在 runtime_pilots 中，不进入正式结果。Llama 的 intrinsic 生成批量为 4，响应级和知识探测批量为 8。',
              '- FD/RU、提取率、误拒代理和覆盖率均为 0–1；原始 quality 为 0–100；耗时单位为秒，显存单位为 GiB。',
              '- 在线 GPU 峰值为 PyTorch max_memory_allocated 的张量已分配量，不包含缓存预留及其他进程；CPU 峰值为主进程 RSS。逐请求记录另保留 reserved 显存。',
              '- FD/RU 仍采用原项目端点插值参考值，属于待显式 retain checkpoint 进一步验证的近似；同时保留原始 Probability/ROUGE/ACC。',
              '- 具体计算：quality = 100 × (Probability × ROUGE-L × ACC)^(1/3)；阶段参考值按请求数/300 在初始与 retain checkpoint 端点之间插值。FD/RU = max(0, 1 − |当前 quality / 对应阶段参考 quality − 1|)，越高表示越接近参考；它们不是直接的知识删除率。',
              '- 单次 seed 的结果不提供跨 seed 显著性结论。在线时间与加载、评估、离线训练时间分开；时间预算在操作边界检查并报告超时量。',
              '- 等预算实验在预算耗尽后接收请求但停止更新，评价集合仍覆盖全部已接收请求；发生更新的请求数与耗尽后的覆盖差异单独报告。','',
              '## P0：Qwen 泛化与过滤消融','',
              '| 方法 | 请求数 | FD | RU | 原始 forget quality | 原始 retain quality | 累计在线秒 | 至该节点在线峰值 GiB |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for result in runs[:7]:
        for stage in result['curves']:
            prefix=result['requests'][:stage['requests']]
            peak=max((row['peak_allocated_gib'] for row in prefix),default=None)
            lines.append(f"| {display_name(result['path'])} | {stage['requests']} | {number(stage.get('FD'))} | {number(stage.get('RU'))} | {number(stage.get('forget',{}).get('quality'))} | {number(stage.get('retain',{}).get('quality'))} | {number(stage.get('online_seconds'))} | {number(peak)} |")
    lines += ['', '## P0：效率与等时间预算','',
              '| 运行 | 在线总秒数 | 每请求中位秒数 | 每请求 P95 秒数 | LM 参数更新请求数 | LM 更新步数 | 在线 GPU 峰值 GiB | CPU 峰值 RSS GiB | 预算超时秒数 |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for result in runs:
        done=result['done'] or {}
        samples=sorted(r['seconds'] for r in result['requests'])
        median=percentile(samples,.5)
        p95=percentile(samples,.95)
        peak=done.get('peak_online_allocated_gib')
        lines.append(f"| {run_name(result['path'])} | {number(done.get('online_seconds'))} | {number(median)} | {number(p95)} | {sum(r.get('updates',0)>0 for r in result['requests'])} | {sum(r.get('updates',0) for r in result['requests'])} | {number(peak)} | {number(done.get('peak_cpu_rss_gib'))} | {number(done.get('budget_overshoot_seconds'))} |")
    lines += ['', '参数更新计数仅指语言模型；CURaTE 更新检索库，未修改语言模型参数，因此该列为 0。',
              '', '### FIT 模块耗时','', '| 运行 | 过滤秒数 | 归因秒数 | 重要性秒数 | 更新秒数 |','|---|---:|---:|---:|---:|']
    for result in runs:
        if result['config'].get('method')!='FIT': continue
        totals={key:sum(r.get('modules',{}).get(key,0) for r in result['requests'])
                for key in ['filter','attribution','importance','update']}
        lines.append(f"| {run_name(result['path'])} | "+' | '.join(number(totals[k]) for k in totals)+' |')
    lines += ['', '### 总流程时间分解','', '| 运行 | 加载秒数 | 预热秒数 | 在线秒数 | 常规评估秒数 | 知识探测秒数 | main 全流程秒数 |','|---|---:|---:|---:|---:|---:|---:|']
    for result in runs:
        done=result['done'] or {}
        timings=done.get('timings',{})
        lines.append(f"| {run_name(result['path'])} | {number(timings.get('load',done.get('load_seconds')))} | {number(timings.get('warmup'))} | {number(done.get('online_seconds'))} | {number(timings.get('evaluation'))} | {number(timings.get('knowledge_probes'))} | {number(done.get('wall_seconds_main'))} |")
    lines += ['', 'main 全流程包含数据、加载、在线处理、评估及可选保存；CURaTE 运行还包含初始模型控制与知识探测，不能将该总量直接视为其在线遗忘开销。未被早期运行记录的分项留空。']
    lines += ['', '### 等预算 FD/RU','', '| 方法 | 请求数 | FD | RU | 在线秒数 |','|---|---:|---:|---:|---:|']
    for result in runs[7:11]:
        for stage in result['curves']:
            lines.append(f"| {display_name(result['path'])} | {stage['requests']} | {number(stage.get('FD'))} | {number(stage.get('RU'))} | {number(stage.get('online_seconds'))} |")
    lines += ['', '## P1：CURaTE 对照','',
              '| 方法 | 请求数 | 目标完整答案泄漏率 | F1≥0.8 提取率 | retain 新增误拒代理 | retain ROUGE-L | retain F1 | retain 答复覆盖率 |',
              '|---|---:|---:|---:|---:|---:|---:|---:|']
    behavior=[]
    for label,data in [('初始模型',corrected_behavior(read(LROOT/'CURaTE/initial_control_evaluations.json',[]),LROOT/'CURaTE',initial=True)),
                       ('CURaTE',runs[-2]['curves']),
                       ('FIT',[r['behavioral'] for r in runs[-1]['curves'] if 'behavioral' in r])]:
        for stage in data:
            f,r=stage['forget'],stage['retain']
            lines.append(f"| {label} | {stage['requests']} | {number(f.get('exact_answer_leakage_rate'))} | {number(f.get('extraction_f1_ge_080_rate'))} | {number(r.get('new_refusal_on_baseline_answerable_proxy'))} | {number(r.get('answer_rougeL'))} | {number(r.get('answer_token_f1'))} | {number(r.get('answer_coverage'))} |")
            behavior.append((label,stage))
    lines += ['', '### 实际在线推理延迟','', '| 方法 | 请求数 | 查询样本数 | 总响应 P50 秒 | 总响应 P95 秒 | 检索/门控平均秒 | 平均生成 token 数 |','|---|---:|---:|---:|---:|---:|---:|']
    for label,stage in behavior:
        latency=stage['latency']
        samples=latency.get('samples',[])
        tokens=sum(row['generated_tokens'] for row in samples)/len(samples) if samples else None
        lines.append(f"| {label} | {stage['requests']} | {latency['all']['n']} | {number(latency['all']['p50_seconds'])} | {number(latency['all']['p95_seconds'])} | {number(latency['gate']['mean_seconds'])} | {number(tokens)} |")
    lines += ['', '以上为真实响应总延迟，包含门控和实际生成；输出长度并未固定，因此不能直接解释为每 token 解码速度。']
    training=read(Path('Model/CURaTE-mpnet-NQ18K-seed42/completed.json'))
    curate_done=runs[-2]['done'] or {}
    if curate_done.get('peak_allocated_gib') is not None:
        lines += ['',f"CURaTE 全流程 GPU 峰值为 {curate_done['peak_allocated_gib']:.3f} GiB；早期运行未单独采样在线峰值，因此在线 GPU 列留空。"]
    if training:
        lines += ['',f"CURaTE 离线句向量模型：{training['data_pairs']} 对官方训练数据，训练 {training['training_seconds']:.2f} 秒，加载 {training['load_seconds']:.2f} 秒，峰值显存 {training['peak_allocated_gib']:.3f} GiB。"]
    lines += ['', '## P1：知识残留检查','', '本节使用 Llama-3-8B，对初始模型、FIT 与 CURaTE 使用同一套问题改写和提示。',
              '', '| 模型/方法 | 目标数 | 任一提示提取率 | 初始可提取目标数 | 初始可提取目标中的残留率 | 初始未提取、后续可提取目标数 |','|---|---:|---:|---:|---:|---:|']
    initial=probes(LROOT/'CURaTE/initial_knowledge_probes.json')
    probe_groups=[]
    for label,path in [('初始模型',LROOT/'CURaTE/initial_knowledge_probes.json'),('CURaTE',LROOT/'CURaTE/knowledge_probes.json'),('FIT',LROOT/'FIT/knowledge_probes.json')]:
        data=probes(path)
        if not data: continue
        probe_groups.append((label,data))
        known=set(initial['extracted_indices']) if initial else set()
        residual=len(known&set(data['extracted_indices']))/len(known) if known else None
        newly_extracted=len(set(data['extracted_indices'])-known) if initial else None
        lines.append(f"| {label} | {data['n_targets']} | {number(data['any_prompt_extraction_rate'])} | {len(known) if initial else '—'} | {number(residual)} | {newly_extracted if newly_extracted is not None else '—'} |")
    lines += ['', '### 按改写与提示形式分组','', '| 方法 | 提示形式 | 查询数 | 完整答案匹配率 | 词面提取率 |','|---|---|---:|---:|---:|']
    for method,data in probe_groups:
        for variant,values in data['by_variant'].items():
            lines.append(f"| {method} | {variant} | {values['n']} | {number(values['exact_rate'])} | {number(values['extraction_rate'])} |")
    bank=read(Path('Experiment_record/probe_bank.json'))
    if bank:
        lines += ['', f"共享改写库：{len(bank['records'])} 条问题，{bank['valid_format']} 条通过自动格式检查。未通过的改写不进入改写分组，但原始问题仍计入总目标数。"]
    lines += ['', '## 复现限制与解释边界','',
              '- ALKN 的官方链接当前返回 404。本轮实现依据论文 Algorithm 1，包括任务向量反向合并；retain 梯度使用固定小批次，全局分位阈值用跨全部参数的固定均匀 131,072 元素样本估计，相关状态卸载至 CPU。此项必须标为论文复现适配。',
              '- O³ 使用官方 OOD 模型损失与 glocal/OCSVM/GMM 门控，LoRA 移植到 Qwen 混合注意力线性层；每次 PCH 请求由当前段落窗口构造 OOD 训练/校准样本。须与作者原数据集上的结果区分。',
              '- O³ 生成评估采用逐样本独立门控的批处理，仅应用接续训练的最新 LoRA，历史 A 只参与正交约束；采用官方 ScienceQA 的推理门控映射与 CPU 协方差估计。评估吞吐不视为单查询推理延迟。',
              '- CURaTE 使用官方 18K 训练数据和默认 0.9 阈值，入库内容仅为当次原始请求；评价问题、答案及未来请求不进入数据库。',
              '- “误拒”目前是初始模型未拒答、处理后拒答的操作性代理，并非人工安全标签；不能把所有合法安全拒答解释为错误。',
              '- 泄漏和知识提取均从逐条生成证据重算；完整答案匹配要求规范化后的词/数字边界，避免将 62 在 16285680 中的出现计为正确答案。知识提取为该匹配或 token F1≥0.8。早期 JSON 中的无边界子串聚合不直接用于本表。',
              '- 上述提取属于词面代理，自动改写尚未经人工语义审核，不能据此证明知识已被彻底删除。',
              '- 注意力后端由每个运行加载后的模块实际记录；Qwen 初始 checkpoint 的准备使用 Torch 路径。正式比较的加速内核状态见各 config.json，不以包已安装推断实际启用。']
    if Path('logs/resource_adjustment.json').exists():
        queue_state=read(Path('logs/all_experiments_state.json'),{})
        budget_gpus=queue_state.get('budget_gpus',{m:queue_state.get('budget_gpu',5) for m in ['FIT','NPO_KL','ALKN','O3']})
        assignments='，'.join(f'{m}→GPU {g}' for m,g in budget_gpus.items())
        lines.append(f'- GPU 1 的外部占用先后约为 23、28 和 59 GiB，曾中断 ALKN/O³ 尝试；未完成组不计为结果。O³ 改用 GPU 4，ALKN 使用 GPU 5。等预算组安排为 {assignments}，各卡同为 H100 NVL。实际物理卡号保存在各 config.json 的 visible_devices 中。')
        if len(set(budget_gpus.values()))>1:
            lines.append('- 用户随后要求利用空卡并行，等预算组因此分布于不同物理 GPU。它们使用相同型号 GPU 和相同在线时间预算，但共享 CPU/内存及其他进程负载可能不同，属于共享环境下的等时间对照，不能称为独占、完全受控的计算公平性实验。')
        lines.append('- 共同在线预算取四个主要方法完整在线耗时的最小值。若未完成方法的累计在线耗时均已超过某个完整方法的最小值，则可提前锁定同一预算并并行启动对照；此时最终最小值不会再改变。')
        if Path('Experiment_record/runtime_pilots/fit_budget_prefix_check.json').exists():
            lines.append('- 等预算组为独立实跑，并非对完整组曲线直接截断。FIT 两组在预算耗尽前已出现选层与分数差异（第 60 请求 FD 为 0.8922/0.9061）；同 seed 未强制 CUDA 逐位确定性，差异原因尚未定位。因此不能把两组的全部指标差异都归因于时间预算，单 seed 也不能估计这种波动。证据见 runtime_pilots/fit_budget_prefix_check.json。')
        if read(Path('logs/resource_adjustment.json')).get('o3_memory_retry'):
            lines.append('- O³ 的旧 GPU 1 尝试在外部进程占用约 28 GiB 时显存不足，未完成组及其阶段指标已归档，不混入当前完整组。历史 B 和暂不用的 FP32 校准矩阵放在 CPU。另发现并修正了早期独立适配器累加的实现错误；相关原型全部归档作废，正式运行使用接续训练的单个最新适配器。')
        if read(Path('logs/resource_adjustment.json')).get('npo_filter_retry'):
            lines.append('- NPO+KL 加过滤的首次试跑在显存分配时失败，启用 expandable_segments 后重试，保持优化器数学计算不变。此前已完成组保留原分配设置；全部等预算组统一采用新分配设置，实际配置见各 config.json。')
    lines += ['', '## 结论状态','']
    lines.append('全部列出的运行均完成。请结合上面的原始指标、实现适配与单 seed 限制解读。' if completed==len(runs)
                 else f'当前仍有 {len(runs)-completed} 项完整运行未完成，尚不能给出覆盖全部实验的最终比较结论。')
    fit=runs[0]
    if fit['complete']:
        lines += ['', '### 已完成对照的数值观察','']
        f=fit['curves'][-1]
        for other in runs[1:4]:
            if not other['complete']: continue
            o=other['curves'][-1]
            ratio=fit['done']['online_seconds']/other['done']['online_seconds'] if other['done']['online_seconds'] else None
            lines.append(f"- Qwen 300 请求：FIT 相对 {display_name(other['path'])} 的 FD 差为 {f['FD']-o['FD']:+.4f}，RU 差为 {f['RU']-o['RU']:+.4f}；在线时间比为 {number(ratio)}。这是单 seed、当前适配实现与指标下的观察。")
    for trial in [runs[3],runs[5],runs[6]]:
        if not trial['complete']: continue
        stage=trial['curves'][-1]
        if stage['forget']['quality']<.1 and stage['retain']['quality']<.1:
            caveat='O³ 为跨架构、跨任务适配，不能据此推出对作者原始设置的一般性结论。' if trial['path'].name=='O3_filter_off' else ''
            lines.append(f"- {display_name(trial['path'])} 在本轮设置下发生 forget/retain quality 同时接近零的整体退化，不能解释为有效的选择性遗忘。{caveat}")
    for label,off,on in [('NPO+KL',runs[1],runs[4]),('RLabel',runs[5],runs[6])]:
        if off['complete'] and on['complete']:
            a,b=off['curves'][-1],on['curves'][-1]
            removed=sum(r.get('deleted_chars',0) for r in on['requests'])
            supplied=sum(r.get('input_chars',0) for r in on['requests'])
            lines.append(f"- 仅加入过滤（{label}）：300 请求处 ΔFD={b['FD']-a['FD']:+.4f}、ΔRU={b['RU']-a['RU']:+.4f}；删除字符比例为 {removed/max(1,supplied):.2%}。此对照未启用选层或重要性路由。")
    if all(trial['complete'] for trial in runs[7:11]):
        bf,bn,ba,bo=[trial['curves'][-1] for trial in runs[7:11]]
        coverage=sum(row['updates']>0 for row in runs[9]['requests'])
        lines.append(f"- 等预算单次实跑：FIT 相对 NPO+KL 的 ΔFD={bf['FD']-bn['FD']:+.4f}、ΔRU={bf['RU']-bn['RU']:+.4f}；相对 ALKN 的 ΔFD={bf['FD']-ba['FD']:+.4f}、ΔRU={bf['RU']-ba['RU']:+.4f}。FD 与 RU 存在取舍，且独立运行波动和共享负载限制了因果解释。")
        lines.append(f"- 等预算 ALKN 仅对 {coverage}/300 条请求执行 LM 更新；其较高 RU 必须结合未更新覆盖率与前述不变模型参考值解读，不能当作完成全部 300 条有效遗忘的证据。O³ 等预算 FD/RU={bo['FD']:.4f}/{bo['RU']:.4f}，须结合整体退化情况解读。")
    if runs[-2]['complete'] and runs[-1]['complete']:
        c=runs[-2]['curves'][-1]
        f=runs[-1]['curves'][-1].get('behavioral')
        if f:
            lines.append(f"- Llama 300 请求：FIT 相对 CURaTE 的完整答案泄漏率差为 {f['forget']['exact_answer_leakage_rate']-c['forget']['exact_answer_leakage_rate']:+.4f}；retain ROUGE-L 差为 {f['retain']['answer_rougeL']-c['retain']['answer_rougeL']:+.4f}。误拒率与延迟应结合前表共同解读，不能仅根据模型权重是否变化判断效果。")
    lines += ['', '## 原始产物与来源','',f'- Qwen：`{QROOT}`',f'- Llama：`{LROOT}`',
              '- 每个运行保留 config.json、requests.jsonl、evaluations.json 和完成标记。',
              '- 核验记录：`Experiment_record/results_audit.json`；包含请求顺序、五个评估节点、数据哈希、数值有限性、计时、共同预算与探测覆盖检查，complete_audited 为已通过核验的组数。',
              '- [CURaTE 论文](https://aclanthology.org/2026.findings-acl.1102/) / [代码](https://github.com/bsu1313/CURaTE)',
              '- [O³ 论文](https://arxiv.org/abs/2407.10223) / [代码](https://github.com/GCYZSL/O3-LLM-UNLEARNING)',
              '- [ALKN 论文](https://arxiv.org/abs/2611.02620)','']
    REPORT.parent.mkdir(parents=True,exist_ok=True)
    temporary=REPORT.with_suffix(f'.md.{os.getpid()}.tmp')
    temporary.write_text('\n'.join(lines))
    temporary.replace(REPORT)
    print(str(REPORT),flush=True)


if __name__=='__main__': main()

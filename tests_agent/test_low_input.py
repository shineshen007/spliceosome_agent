"""Local RDS discovery, concise confirmation and optional candidate screening."""
from pathlib import Path
import json
import subprocess

import numpy as np
import pandas as pd
import pytest

from spliceosome_agent.data import child_environment, inspect_assay, resource
from spliceosome_agent.discovery import propose_inputs
from spliceosome_agent.engine import Engine, verify_run
from spliceosome_agent.io import object_hash, read_json, sha256
from spliceosome_agent.models import StudySpec
from spliceosome_agent.planning import input_paths, load_study
from spliceosome_agent.runner import execute_run
from spliceosome_agent.session import Session


@pytest.fixture
def rds_input(tmp_path, study):
    folder = tmp_path / 'rds'
    folder.mkdir()
    expression = pd.read_csv(study.assays['protein'].matrix, index_col=0)
    expression.to_csv(folder/'protein.csv')
    md = pd.read_csv(study.assays['protein'].metadata)
    md = md.rename(columns={'sample_id':'ID','group':'Type','subject_id':'Pair'})[['ID','Type','Pair']]
    md.to_csv(folder/'metadata.csv',index=False)
    script = '''args<-commandArgs(TRUE); x<-read.csv(args[1],row.names=1,check.names=FALSE);
    m<-read.csv(args[2],check.names=FALSE); saveRDS(x,args[3]); saveRDS(m,args[4])'''
    result=subprocess.run(['Rscript','--vanilla','-e',script,str(folder/'protein.csv'),str(folder/'metadata.csv'),str(folder/'protein_expr.rds'),str(folder/'protein_info.rds')],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    return folder


def auto_session(tmp_path, folder):
    s=Session(tmp_path/'workspace',tmp_path/'models.json')
    s._register_files([str(folder/'protein_expr.rds'),str(folder/'protein_info.rds')])
    return s


@pytest.fixture
def quick_environment(monkeypatch):
    monkeypatch.setattr('spliceosome_agent.preflight.check_environment',lambda study: ({'test':'stable'},[]))
    monkeypatch.setattr('spliceosome_agent.engine.r_fingerprint',lambda study: {'test':'stable'})


def test_rds_is_inspected_before_full_study_and_only_scale_is_asked(tmp_path,rds_input):
    s=auto_session(tmp_path,rds_input)
    class NoQuestionnaire:
        def json(self,*args,**kwargs): pytest.fail('clear local inputs must not generate a model questionnaire')
    s.client=NoQuestionnaire()
    # Restrict explicit scope to the two RDS files, just as in the real screenshot.
    text=f'使用 "{rds_input/"protein_expr.rds"}" 和 "{rds_input/"protein_info.rds"}" 做蛋白差异分析并画火山图'
    reply=s.converse(text)
    card=s.store.get('confirmation_card')
    assert card['status']=='needs_scale'
    assert card['n_features']==160 and card['n_samples']==24 and card['n_units']==12
    assert card['study']['design']=='paired'
    assert card['study']['assays']['protein']['columns']=={'sample_id':'ID','group':'Type','subject_id':'Pair'}
    assert 'Confirm measurement scale' in reply
    assert all(v not in reply for v in ['missing','癌栓','file_000','物种是什么','列名是什么'])
    assert s.store.get('current_plan') is None
    assert not list(s.store.workspace.rglob('run_analysis.py'))
    restored=Session(s.store.workspace,s.config_path)
    assert restored.store.get('confirmation_card')==card


def test_scale_confirmation_makes_plain_plan_and_legacy_hash_stays_stable(tmp_path,rds_input,quick_environment,study):
    s=auto_session(tmp_path,rds_input)
    s.auto_prepare('蛋白差异分析，火山图')
    reply=s.choose_scale('log2')
    plan=s.current_plan()
    assert 'Confirm and run' in reply
    assert plan.study.schema_version=='2.0' and plan.study.analysis_mode=='differential'
    assert plan.study.species=='unknown' and plan.study.candidates is None
    assert plan.study.assays['protein'].id_type=='UNKNOWN'
    assert [t.role for t in plan.tasks]==['QC','Protein','Critic']
    assert set(input_paths(plan.study))=={str(rds_input/'protein_expr.rds'),str(rds_input/'protein_info.rds')}
    assert plan.preflight.summary['_field_sources']['scale']['status']=='user_confirmed'
    assert not s.store.is_approved(object_hash(plan))
    assert not list(s.store.workspace.rglob('run_analysis.py'))
    legacy=study.model_dump(mode='json')
    assert 'analysis_mode' not in legacy
    assert all('transform' not in a for a in legacy['assays'].values())
    assert object_hash(legacy)==object_hash(StudySpec.model_validate(legacy))


def test_unknown_scale_or_input_drift_never_approves(tmp_path,rds_input,quick_environment):
    s=auto_session(tmp_path,rds_input)
    s.auto_prepare('蛋白差异分析')
    s.choose_scale('log2')
    digest=s.store.get('current_plan')
    s.store.approve(digest)
    assert 'will not run' in s.choose_scale('unknown')
    with pytest.raises(PermissionError): s.confirm_and_run(digest)
    with pytest.raises(PermissionError): s.store.require_approval(digest)
    p=rds_input/'protein_info.rds'
    p.write_bytes(p.read_bytes()+b'\n')
    with pytest.raises(ValueError,match='Input changed'): s.choose_scale('log2')
    assert not list(s.store.workspace.rglob('run_analysis.py'))


def test_pair_conflicts_and_unobserved_thrombus_are_not_guessed(tmp_path,rds_input):
    s=auto_session(tmp_path,rds_input)
    reply=s.auto_prepare('按癌栓进行蛋白差异分析')
    assert 'No thrombus field was found' in reply
    assert s.store.get('current_plan') is None
    command='args<-commandArgs(TRUE);m<-readRDS(args[1]);m$Pair[1]<-"different";saveRDS(m,args[1])'
    subprocess.run(['Rscript','--vanilla','-e',command,str(rds_input/'protein_info.rds')],check=True)
    reply=s.auto_prepare('蛋白差异分析')
    assert 'cannot form one-to-one pairs' in reply
    assert s.store.get('confirmation_card')['status']=='blocked'


def test_multiple_inputs_offer_dataset_choice(tmp_path,rds_input):
    second=rds_input/'second_protein.rds'
    second.write_bytes((rds_input/'protein_expr.rds').read_bytes())
    s=auto_session(tmp_path,rds_input)
    s._register_files([str(second)])
    reply=s.auto_prepare('蛋白差异分析')
    assert 'Multiple datasets found' in reply
    assert len(s.store.get('confirmation_card')['alternatives'])==2
    reply=s.select_dataset(0)
    assert 'Confirm measurement scale' in reply
    assert s.store.get('confirmation_card')['status']=='needs_scale'


def test_swap_changes_only_direction_and_requires_new_confirmation(tmp_path,rds_input,quick_environment):
    s=auto_session(tmp_path,rds_input)
    s.auto_prepare('蛋白差异分析');s.choose_scale('log2')
    old=s.current_plan();digest=object_hash(old);s.store.approve(digest)
    s.swap_direction()
    new=s.current_plan()
    assert (new.study.case,new.study.reference)==(old.study.reference,old.study.case)
    assert object_hash(new)!=digest
    assert not s.store.is_approved(digest)
    assert new.preflight.summary['_field_sources']['contrast']['status']=='user_confirmed'


def test_explicit_rds_scale_attribute_does_not_need_a_question(tmp_path,rds_input,quick_environment):
    command='args<-commandArgs(TRUE);x<-readRDS(args[1]);attr(x,"scale")<-"log2_abundance";saveRDS(x,args[1])'
    subprocess.run(['Rscript','--vanilla','-e',command,str(rds_input/'protein_expr.rds')],check=True)
    s=auto_session(tmp_path,rds_input)
    reply=s.auto_prepare('蛋白差异分析')
    assert 'Confirm measurement scale' not in reply
    assert s.current_plan().study.assays['protein'].scale=='log2_abundance'
    assert s.current_plan().preflight.summary['_field_sources']['scale']['status']=='observed'


@pytest.mark.parametrize('choice',['log2','raw_log2'])
def test_plain_differential_real_r_and_volcano_without_candidates(tmp_path,rds_input,choice):
    s=auto_session(tmp_path,rds_input)
    s.auto_prepare('蛋白差异分析，火山图');s.choose_scale(choice)
    plan=s.current_plan();digest=object_hash(plan)
    inputs={p:sha256(p) for p in plan.input_hashes}
    s.store.approve(digest)
    engine=Engine.create(s.store,digest)
    state=execute_run(engine)
    assert state['status']=='complete',state
    assert set(state['results'])=={'QC','Protein','Critic'}
    assert not (engine.run/'knowledge').exists() and not (engine.run/'integration').exists()
    assert (engine.run/'protein/differential.csv').exists()
    for ext in ['pdf','svg']: assert (engine.run/f'figures/protein_volcano.{ext}').exists()
    assert (engine.run/'figures/source_data/protein_differential.csv').exists()
    assert not list(engine.run.rglob('*.png'))
    assert verify_run(s.store,engine.run_id)['status']=='verified'
    observed=pd.read_csv(engine.run/'inputs/protein/matrix.csv',index_col=0)
    original=pd.read_csv(rds_input/'protein.csv',index_col=0)
    target=np.log2(original) if choice=='raw_log2' else original
    assert np.allclose(observed.values,target.values)
    assert inputs=={p:sha256(p) for p in inputs}
    # Independent reference checks use the actual frozen job, not implementation-shaped mocks.
    reference=tmp_path/'reference.csv'
    response=subprocess.run(['Rscript','--vanilla',str(Path(__file__).parent/'reference_models.R'),str(engine.run/'protein/job.json'),str(reference)],
                            env=child_environment(plan.study),capture_output=True,text=True)
    assert response.returncode==0,response.stderr
    a=pd.read_csv(engine.run/'protein/differential.csv').set_index('feature_id');b=pd.read_csv(reference).set_index('feature_id')
    for column in ['effect','se','pvalue','padj','ci_low','ci_high']:
        assert np.allclose(a.loc[b.index,column],b[column],rtol=2e-7,atol=1e-9,equal_nan=True)


def test_nonpositive_abundance_never_gets_an_implicit_pseudocount(tmp_path,rds_input):
    command='args<-commandArgs(TRUE);x<-readRDS(args[1]);x[1,1]<-0;saveRDS(x,args[1])'
    subprocess.run(['Rscript','--vanilla','-e',command,str(rds_input/'protein_expr.rds')],check=True)
    s=auto_session(tmp_path,rds_input)
    s.auto_prepare('蛋白差异分析')
    with pytest.raises(ValueError,match='No pseudocount will be added'):s.choose_scale('raw_log2')
    assert s.store.get('current_plan') is None
    assert not list(s.store.workspace.rglob('run_analysis.py'))


def test_rna_raw_log2_choice_is_a_valid_plain_study(study):
    payload=study.model_dump()
    payload.update(schema_version='2.0',analysis_mode='differential',candidates=None,species='unknown')
    payload['assays']={'rna':payload['assays']['rna']}
    payload['assays']['rna'].update(scale='abundance',transform='log2',id_type='UNKNOWN')
    plain=StudySpec.model_validate(payload)
    assert plain.assays['rna'].transform=='log2'


def test_first_model_request_already_has_rds_probes_without_private_ids(tmp_path,rds_input):
    import httpx
    from spliceosome_agent.config import ModelConfig,ModelProfile
    from spliceosome_agent.llm import ModelClient
    seen=[]
    def reply(request):
        seen.append(request.content.decode())
        context=json.loads(json.loads(request.content)['messages'][1]['content'])
        assert len(context['probe_results'])>=2
        assert any(p.get('declared_columns') and 'ID' in p['declared_columns'] for p in context['probe_results'].values())
        return httpx.Response(200,json={'choices':[{'message':{'content':json.dumps({'explanation':'Please confirm the intended purpose.'})}}], 'usage':{'total_tokens':10}})
    config=ModelConfig(default='mock',profiles={'mock':ModelProfile(name='mock',model='mock',kind='local',base_url='http://127.0.0.1:9999/v1',api_key_env=None)})
    s=auto_session(tmp_path,rds_input)
    s.client=ModelClient(config,tmp_path/'audit',transport=httpx.MockTransport(reply))
    s.converse('Question about SYNTH_SUBJECT_00 in these files')
    assert len(seen)==1
    assert 'SYNTH_SUBJECT_00' not in seen[0] and 'SYNTH_CONTROL_00' not in seen[0]
    assert 'file_000' in seen[0]
    assert str(rds_input) not in seen[0]

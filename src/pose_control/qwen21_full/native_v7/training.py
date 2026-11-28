"""V7 configuration and objectives on the shared, strict V6.2 training engine."""
from dataclasses import dataclass, field, replace
from typing import ClassVar

from ..native_v62.training.config import TrainingConfig, stage_preset
from ..native_v62.training.trainer import NativeV62Trainer
from ..native_v62.training.backend import NativeBackend
from ..native_v62.training.data import DataPlan
from .losses import PaperLossConfig

SCHEMA = 'native_v7_training_v1'


def stage_preset_v7(phase, branch=None, *, budget=None):
    stage = stage_preset(phase, branch, budget=budget)
    return replace(stage, routes=tuple(replace(route, loss=replace(route.loss, identity=.1)) for route in stage.routes))


@dataclass(frozen=True)
class TrainingConfigV7(TrainingConfig):
    expected_schema: ClassVar[str] = SCHEMA
    schema: str = SCHEMA
    qwen_condition_profile: str = 'dual_4ref'
    stage: object = field(default_factory=lambda: stage_preset_v7('S1', 'poselink'))
    paper_loss: PaperLossConfig = field(default_factory=PaperLossConfig)
    preservation: object | None = None
    preservation_index: dict | None = None

    def __post_init__(self):
        if isinstance(self.paper_loss, dict):
            object.__setattr__(self, 'paper_loss', PaperLossConfig.from_dict(self.paper_loss))
        if not isinstance(self.paper_loss, PaperLossConfig):
            raise ValueError('Explicit PaperLossConfig required')
        from .preservation import PreservationConfig
        from ..native_v62.training.config import reference_syntax
        if isinstance(self.preservation, dict):
            object.__setattr__(self, 'preservation', PreservationConfig.from_dict(self.preservation))
        if self.preservation is not None and not isinstance(self.preservation, PreservationConfig):
            raise ValueError('Explicit PreservationConfig required')
        if self.preservation is None and self.preservation_index is not None:
            raise ValueError('Preservation index requires an enabled objective')
        reference_syntax(self.preservation_index)
        super().__post_init__()
        if any(getattr(r.loss,'region_mode','legacy') != 'legacy' for r in self.stage.routes) and (self.paper_loss.person_fm or self.paper_loss.contact_fm):
            raise ValueError('Regional FM replaces V7 additive Person/Contact FM; set both explicitly to zero')

    @property
    def additional_evaluators(self):
        return ('identity',) if (self.paper_loss.identity_contrastive or
            self.preservation is not None and self.preservation.source_identity) else ()

    def record(self):
        result = super().record()
        if self.preservation is None:
            result.pop('preservation'); result.pop('preservation_index')
        return result

    def missing(self):
        result = super().missing()
        identity = self.additional_evaluators or any(route.loss.identity for route in self.stage.routes)
        if identity and (not self.evaluators or not self.evaluators.get('identity')):
            result.append('evaluators.identity')
        if self.preservation is not None and not self.preservation_index:
            result.append('preservation_index')
        return result


def validate_paper_eligibility(rows, config):
    selected = [row for row in rows if row['eligible']]
    qualified = sum(row['auxiliary_faces'] >= 2 for row in selected)
    if config.identity_contrastive and not qualified:
        raise ValueError('Requested identity contrastive loss has no qualified dual-person pair; '
                         'export valid GT five-point supervision or explicitly disable identity_contrastive')
    return {'eligible_rows': len(selected), 'contrastive_pairs': qualified,
            'qualified_faces': sum(row['auxiliary_faces'] for row in selected)}


def paper_sample_status(sample):
    import torch
    from ..native_v62.conditions import part_contact_permissions
    local=sample['local_condition_v4']
    status={'paper_persons':int(local.person_valid.sum()),'paper_positive_contacts':0,
            'paper_forbidden_contacts':0,'paper_visible_contact_persons':0,
            'paper_repair_supplied':'pose_repair_v61' in sample}
    depth=sample.get('dual_depth')
    if depth is None:
        return status
    positive,forbidden=part_contact_permissions(depth.part_contact)
    active=torch.stack((positive.any(2),positive.any(1)),1)
    known_parts=torch.stack((depth.person_a.context.part_valid,depth.person_b.context.part_valid),1)
    height,width=depth.person_part.shape[-2:]
    valid=sample['valid_image_mask'].to(depth.person_part.device).reshape(1,1,height,width)
    parts=depth.person_part*known_parts*valid[:,None]
    support=(parts*active[...,None,None]).sum(2)>0
    owner=(depth.owner_weights.squeeze(2)>0)&(valid>0)
    visible=(support&owner).flatten(2).any(-1)&local.person_valid.to(owner.device)
    return {**status,'paper_positive_contacts':int(positive.sum()),
            'paper_forbidden_contacts':int(forbidden.sum()),'paper_visible_contact_persons':int(visible.sum())}


class DataPlanV7(DataPlan):
    def sample_metadata(self,sample):
        return paper_sample_status(sample)

    def __init__(self, config):
        super().__init__(config)
        self.paper_eligibility = {}
        for route in config.stage.routes:
            selected = [row for row in self.records if row['role'] == 'train' and row['route'] == route.name]
            self.paper_eligibility[route.name] = validate_paper_eligibility(selected, config.paper_loss)
        self.preservation_records = {}
        self.preservation_eligibility = {}
        if config.preservation is not None:
            self._bind_preservation()

    def _bind_preservation(self):
        from .preservation_assets import read_index, read_record
        from ..native_v62.training.config import digest
        from ..native_v62.training.io import verify
        index, bound = read_index(self.config.preservation_index)
        evaluator_sha = index['evaluator']['sha256']
        if self.config.preservation.source_identity:
            if verify((self.config.evaluators or {}).get('identity'))['sha256'] != evaluator_sha:
                raise ValueError('Source identity evaluator differs from preservation features')
        expected = {source+'\0'+row['pair_id'] for source,dataset in self.datasets.items() for row in dataset.rows}
        if set(index['entries']) != expected:
            raise ValueError('Preservation index must bind exactly the declared samples')
        for source,dataset in self.datasets.items():
            for i,row in enumerate(dataset.rows):
                sample = self.sample(source,i)
                record = read_record(index['entries'][source+'\0'+row['pair_id']],sample,evaluator_sha)
                self.preservation_records[(source,i)] = record
                faces = int((record['source_valid'] & sample['training_supervision_v5'].auxiliary_valid.flatten().cpu()).sum())
                background = bool(record['background_support'].sum()>0)
                for route in self.config.stage.routes:
                    self.status[(route.name,source,i)].update(preservation_faces=faces,preservation_background=background)
                for status in self.records:
                    if status['source_id']==source and status['index']==i:
                        status.update(preservation_faces=faces,preservation_background=background)
        for route in self.config.stage.routes:
            rows = [r for r in self.records if r['role']=='train' and r['route']==route.name and r['eligible']]
            faces = sum(r['preservation_faces'] for r in rows)
            background = sum(r['preservation_background'] for r in rows)
            if self.config.preservation.source_identity and not faces:
                raise ValueError('Requested Source identity has no qualified Source/GT face intersection: '+route.name)
            if self.config.preservation.background and not background:
                raise ValueError('Requested Source background has no eligible support: '+route.name)
            self.preservation_eligibility[route.name] = {'faces':faces,'background_images':background}
        self.identity['preservation'] = {'index':bound,'evaluator':index['evaluator'],'entries':index['entries']}
        self.signature = digest(self.identity)

    def report(self):
        rows = [row for dataset in self.datasets.values() for row in dataset.rows]
        return {**super().report(), 'schema': 'native_v7_data_preflight_v1',
                'paper_eligibility': self.paper_eligibility,
                'geometry_qualification':{
                    'dual_rows':sum(row['paper_persons']==2 for row in self.records if row['route']==self.config.stage.routes[0].name),
                    'visible_contact_rows':sum(row['paper_visible_contact_persons']==2 for row in self.records if row['route']==self.config.stage.routes[0].name)},
                'source_repair': {'manifest_rows': len(rows), 'supplied': sum(bool(row.get('pose_repair_v61')) for row in rows)},
                'paper_loss': self.config.paper_loss.to_dict(),
                **({'preservation_eligibility':self.preservation_eligibility} if self.config.preservation else {})}


class NativeBackendV7(NativeBackend):
    def gradient_case_key(self,status):
        key = (status['auxiliary_faces'],status.get('paper_positive_contacts',0)>0,
                status.get('paper_visible_contact_persons',0),status.get('paper_repair_supplied',False))
        return key + ((status.get('preservation_faces',0),status.get('preservation_background',False))
                      if self.config.preservation is not None else ())

    @property
    def installer(self):
        from .integration import install_native_full_v7
        return install_native_full_v7

    @property
    def runtime_type(self):
        from .runtime import NativeFullRuntimeV7
        return NativeFullRuntimeV7

    @property
    def extra_auxiliary(self):
        return bool(self.config.paper_loss.identity_contrastive or
                    self.config.preservation is not None and self.config.preservation.source_identity)

    def pair(self,source,index):
        sample,batch,pair = super().pair(source,index)
        if self.config.preservation is not None:
            batch['source_preservation_v7'] = self.data.preservation_records[(source,index)]
        return sample,batch,pair

    def forward(self,batch,pair,route,sigma,noise,options,base):
        result = super().forward(batch,pair,route,sigma,noise,options,base)
        self.preservation_context = None
        if self.config.preservation is not None:
            record = batch['source_preservation_v7']
            has_background = not bool((record['background_support'] == 0).all())
            if self.config.preservation.background and has_background and pair['source_latent'].shape != pair['target_latent'].shape:
                raise ValueError('Source background requires exactly the target latent grid')
            self.preservation_context = {name:record[name].to(self.device).detach()
                for name in ('source_features','source_valid','background_support')}
            self.preservation_context['source_latent'] = pair['source_latent'].detach()
        return result

    def prepare_loss_context(self, prepared):
        import torch
        routing = prepared.routing
        positive = routing.contact_positive & ~routing.contact_forbidden
        active_parts = torch.stack((positive.any(2), positive.any(1)), 1)
        contact = (routing.part_tokens * active_parts[:, :, None]).sum(-1).transpose(1, 2)
        # These are target-Canvas conditions, not learned outputs or supervision
        # inferred from Target RGB. Detach so the encoder cannot shrink its loss.
        return {'owner': prepared.pose.owner.detach(), 'contact': contact.detach().clamp(0, 1)}

    def window_counts(self, entries, routes, update):
        result = super().window_counts(entries, routes, update)
        result['paper_identity'] = 0
        if self.config.preservation is not None:
            result.update(source_identity=0,source_background=0)
            for entry in entries:
                spec=routes[entry['route']].loss
                status=entry['status']
                if self.config.preservation.source_identity and update%spec.auxiliary_every==0 and entry['sigma']<=spec.auxiliary_sigma_max:
                    result['source_identity'] += status['preservation_faces']
                if self.config.preservation.background:
                    result['source_background'] += int(status['preservation_background'])
        if self.config.paper_loss.identity_contrastive:
            for entry in entries:
                loss = routes[entry['route']].loss
                faces = entry['status']['auxiliary_faces']
                if update % loss.auxiliary_every == 0 and entry['sigma'] <= loss.auxiliary_sigma_max and faces >= 2:
                    result['paper_identity'] += faces
        return result

    def compute_loss(self, prediction, target, coverage, supervision, noisy, sigma, target_rgb,
                     loss_spec, counts, update, evaluators, *, spatial=None):
        import torch
        from .losses import aligned_identity_regularizer, paper_region_losses
        paper = self.config.paper_loss
        callback = aligned_identity_regularizer(paper.temperature) if counts.get('paper_identity', 0) else None
        preservation = self.config.preservation
        if preservation is not None and counts.get('source_identity',0):
            from .preservation import source_identity_loss
            previous_callback = callback
            context = self.preservation_context
            def callback(generated,references,slots):
                result = previous_callback(generated,references,slots) if previous_callback is not None else {}
                if any(row!=0 or person not in (0,1) for row,person in slots) or len(set(slots))!=len(slots):
                    raise ValueError('Source identity callback requires unique B1 person slots')
                positions = {person:i for i,(_,person) in enumerate(slots)}
                values = [generated[positions[p]] if p in positions else generated.new_zeros(generated.shape[-1]) for p in (0,1)]
                valid = context['source_valid'].to(generated.device) & torch.tensor([p in positions for p in (0,1)],device=generated.device)
                value,count = source_identity_loss(torch.stack(values)[None],context['source_features'].to(generated.device)[None],valid[None])
                return {**result,'source_identity':value,'source_identity_eligible':count}
        if callback is not None and (evaluators is None or evaluators[0] is None):
            evaluators = self.ensure_evaluators({'identity'})
        loss, parts = super().compute_loss(prediction, target, coverage, supervision, noisy, sigma,
            target_rgb, loss_spec, counts, update, evaluators, spatial=spatial, identity_regularizer=callback)
        if paper.person_fm or paper.contact_fm:
            if not isinstance(getattr(self, 'loss_context', None), dict):
                raise ValueError('V7 objective requires the same forward prepared geometry')
            regions = paper_region_losses(prediction, target, coverage, **self.loss_context)
            loss = loss + (paper.person_fm * regions['person_fm'] + paper.contact_fm * regions['contact_fm']) / counts['velocity']
            parts.update(regions)
        eligible = parts.get('contrastive_eligible', 0)
        if counts.get('paper_identity', 0) and eligible:
            loss = loss + paper.identity_contrastive * parts['identity_contrastive'] * eligible / counts['paper_identity']
        if preservation is not None:
            count = parts.get('source_identity_eligible',0)
            if counts.get('source_identity',0) and count:
                loss = loss + preservation.source_identity * parts['source_identity'] * count / counts['source_identity']
            if preservation.background:
                from .preservation import source_background_loss
                context=self.preservation_context
                support=context['background_support']
                source_latent=context['source_latent']
                if bool((support == 0).all()) and source_latent.shape != prediction.shape:
                    source_latent=prediction.detach()
                value,area=source_background_loss(prediction,noisy,sigma,source_latent,support,coverage)
                parts.update(source_background=value,source_background_area=area)
                if area>0:
                    if not counts.get('source_background',0):
                        raise ValueError('Source background window denominator is empty')
                    loss=loss+preservation.background*value/counts['source_background']
        parts['weighted_micro_objective'] = loss
        return loss, parts


class NativeV7Trainer(NativeV62Trainer):
    config_type = TrainingConfigV7
    backend_type = NativeBackendV7
    data_plan_type = DataPlanV7


def main(argv=None):
    from ..native_v62.training.cli import main as shared_main
    return shared_main(argv, config_type=TrainingConfigV7, trainer_type=NativeV7Trainer, stage_factory=stage_preset_v7)

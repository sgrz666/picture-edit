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

    def __post_init__(self):
        if isinstance(self.paper_loss, dict):
            object.__setattr__(self, 'paper_loss', PaperLossConfig.from_dict(self.paper_loss))
        if not isinstance(self.paper_loss, PaperLossConfig):
            raise ValueError('Explicit PaperLossConfig required')
        super().__post_init__()
        if any(r.loss.region_mode != 'legacy' for r in self.stage.routes) and (self.paper_loss.person_fm or self.paper_loss.contact_fm):
            raise ValueError('Regional FM replaces V7 additive Person/Contact FM; set both explicitly to zero')

    @property
    def additional_evaluators(self):
        return ('identity',) if self.paper_loss.identity_contrastive else ()

    def missing(self):
        result = super().missing()
        identity = self.paper_loss.identity_contrastive or any(route.loss.identity for route in self.stage.routes)
        if identity and (not self.evaluators or not self.evaluators.get('identity')):
            result.append('evaluators.identity')
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

    def report(self):
        rows = [row for dataset in self.datasets.values() for row in dataset.rows]
        return {**super().report(), 'schema': 'native_v7_data_preflight_v1',
                'paper_eligibility': self.paper_eligibility,
                'geometry_qualification':{
                    'dual_rows':sum(row['paper_persons']==2 for row in self.records if row['route']==self.config.stage.routes[0].name),
                    'visible_contact_rows':sum(row['paper_visible_contact_persons']==2 for row in self.records if row['route']==self.config.stage.routes[0].name)},
                'source_repair': {'manifest_rows': len(rows), 'supplied': sum(bool(row.get('pose_repair_v61')) for row in rows)},
                'paper_loss': self.config.paper_loss.to_dict()}


class NativeBackendV7(NativeBackend):
    def gradient_case_key(self,status):
        return (status['auxiliary_faces'],status.get('paper_positive_contacts',0)>0,
                status.get('paper_visible_contact_persons',0),status.get('paper_repair_supplied',False))

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
        return self.config.paper_loss.identity_contrastive > 0

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
        if self.config.paper_loss.identity_contrastive:
            for entry in entries:
                loss = routes[entry['route']].loss
                faces = entry['status']['auxiliary_faces']
                if update % loss.auxiliary_every == 0 and entry['sigma'] <= loss.auxiliary_sigma_max and faces >= 2:
                    result['paper_identity'] += faces
        return result

    def compute_loss(self, prediction, target, coverage, supervision, noisy, sigma, target_rgb,
                     loss_spec, counts, update, evaluators, *, spatial=None):
        from .losses import aligned_identity_regularizer, paper_region_losses
        paper = self.config.paper_loss
        callback = aligned_identity_regularizer(paper.temperature) if counts.get('paper_identity', 0) else None
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
        parts['weighted_micro_objective'] = loss
        return loss, parts


class NativeV7Trainer(NativeV62Trainer):
    config_type = TrainingConfigV7
    backend_type = NativeBackendV7
    data_plan_type = DataPlanV7


def main(argv=None):
    from ..native_v62.training.cli import main as shared_main
    return shared_main(argv, config_type=TrainingConfigV7, trainer_type=NativeV7Trainer, stage_factory=stage_preset_v7)

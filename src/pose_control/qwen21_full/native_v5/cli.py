"""Shared V5 CLI options; importing this module does not import a model."""


def add_controls(p):
    p.add_argument('--data-profile',choices=('auto','full_smplx','hybrid_hi4d'),default='auto')
    p.add_argument('--depth-mode',choices=('off','scene_only','full_geometry'),default='full_geometry')
    for flag in ('face','hand','source-detail','depth-detail','interaction','raster','part-contact','body-identity'):
        p.add_argument('--no-'+flag,action='store_true')
    for name in ('body','interaction','detail','face','hand','depth','relation','body-identity'):
        p.add_argument('--'+name+'-strength',type=float,default=1.)
    for name in ('face-identity','face-texture','face-geometry','hand-geometry','hand-texture'):
        p.add_argument('--'+name+'-gain',type=float,default=1.)


def options_for(args,kind):
    from .contracts import NativeControlOptionsV5
    from ..native_dual_depth import DualControlOptions
    local=NativeControlOptionsV5(face=not args.no_face,hand=not args.no_hand,detail=not args.no_source_detail,
        interaction=not args.no_interaction and kind=='dual',body_strength=args.body_strength,
        interaction_strength=args.interaction_strength,detail_strength=args.detail_strength,
        face_strength=args.face_strength,hand_strength=args.hand_strength,
        face_identity_gain=args.face_identity_gain,face_texture_gain=args.face_texture_gain,face_geometry_gain=args.face_geometry_gain,
        hand_geometry_gain=args.hand_geometry_gain,hand_texture_gain=args.hand_texture_gain)
    physical=args.depth_mode=='full_geometry';interaction=kind=='dual' and not args.no_interaction
    base=DualControlOptions(depth_mode=args.depth_mode,detail=physical and not args.no_depth_detail,
        interaction=interaction,raster=interaction and physical and not args.no_raster,
        part_contact=interaction and physical and not args.no_part_contact,identity=not args.no_body_identity,
        depth_strength=args.depth_strength,relation_strength=args.relation_strength,
        interaction_strength=1.,identity_strength=args.body_identity_strength)
    return local,base


def profile_controls(profile,options):
    from dataclasses import replace
    # Explicit profiles select effective branches, not an automatic phase scheduler.
    if profile.startswith('face-'): return replace(options,hand=False,detail=False,interaction=False)
    if profile.startswith('hand-'): return replace(options,face=False,detail=False,interaction=False)
    if profile.startswith('detail-'): return replace(options,interaction=False,face_identity_gain=0.,face_texture_gain=0.,face_geometry_gain=0.,hand_geometry_gain=0.,hand_texture_gain=0.)
    if profile.startswith('interaction-'): return replace(options,face=False,hand=False,detail=False)
    return options

"""Exact triangle surface queries using an isolated python-fcl installation."""
from pathlib import Path
import sys
import numpy as np

DEFAULT_FCL_ROOT='/data/shangguanrz/pic-edit-workflow-v2/deps/python_fcl'

def ensure_fcl(root=None):
    root=Path(root or DEFAULT_FCL_ROOT)
    if root.is_dir() and str(root) not in sys.path:sys.path.insert(0,str(root))
    import fcl
    return fcl

def mesh_collision(a,b):
    fcl=ensure_fcl()
    def object_for(person):
        vertices=np.asarray(person.get('vertices_cam',person.get('mesh_cam')),np.float64)
        faces=np.asarray(person['faces'],np.int32)
        model=fcl.BVHModel();model.beginModel(len(vertices),len(faces));model.addSubModel(vertices,faces);model.endModel()
        return fcl.CollisionObject(model,fcl.Transform())
    first,second=object_for(a),object_for(b)
    result=fcl.CollisionResult();fcl.collide(first,second,fcl.CollisionRequest(num_max_contacts=2048,enable_contact=True),result)
    if result.is_collision:distance=0.
    else:distance=float(fcl.distance(first,second,fcl.DistanceRequest(),fcl.DistanceResult()))
    return dict(intersects=bool(result.is_collision),triangle_contacts=len(result.contacts),
        contacts_capped=len(result.contacts)>=2048,surface_distance_m=max(0.,distance),
        method='FCL_triangle_surface; contact_count_is_not_penetration_volume')

def safe_contact_scale(count):
    before=int(count(0.))
    for scale in (1.,.75,.5,.25,0.):
        after=int(count(scale))
        if after<=before:
            return scale,dict(before_triangle_contacts=before,after_triangle_contacts=after,
                accepted_contact_step_scale=scale,rule='do_not_increase_triangle_intersections')
    raise RuntimeError('Original contact state must remain a feasible fallback')

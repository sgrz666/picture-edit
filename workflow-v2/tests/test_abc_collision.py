"""Real triangle collision queries and conservative contact correction."""
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

def tetra(offset):
    return {'vertices_cam':np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],np.float64)+offset,
        'faces':np.array([[0,2,1],[0,1,3],[0,3,2],[1,2,3]],np.int32)}

def test_collision_queries_distinguish_separated_and_intersecting_meshes():
    from retarget.collision import mesh_collision
    separated=mesh_collision(tetra(np.array([0,0,2])),tetra(np.array([3,0,2])))
    assert not separated['intersects'] and separated['surface_distance_m']>1
    intersecting=mesh_collision(tetra(np.array([0,0,2])),tetra(np.array([.1,.1,2.1])))
    assert intersecting['intersects'] and intersecting['triangle_contacts']>0

def test_contact_step_is_rejected_when_it_increases_intersections():
    from retarget.collision import safe_contact_scale
    # A proposed shift can reduce hand distance while pushing the torsos through
    # each other. Keep the original solution if every positive step is worse.
    scale,diagnostics=safe_contact_scale(lambda scale:3 if scale==0 else 5)
    assert scale==0 and diagnostics['after_triangle_contacts']==3

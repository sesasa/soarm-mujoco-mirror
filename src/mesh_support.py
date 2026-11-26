"""Cache convex-hull vertices for exact mesh support and projected hull checks."""
from functools import lru_cache
import numpy as np
from scipy.spatial import ConvexHull,QhullError

@lru_cache(maxsize=8)
def mesh_hulls(model):
    output=[]
    for mesh in range(model.nmesh):
        start=int(model.mesh_vertadr[mesh]);count=int(model.mesh_vertnum[mesh])
        vertices=np.asarray(model.mesh_vert[start:start+count],dtype=float)
        try:vertices=vertices[ConvexHull(vertices).vertices]
        except QhullError:pass
        vertices.setflags(write=False);output.append(vertices)
    return tuple(output)

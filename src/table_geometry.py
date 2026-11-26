"""Extrude a simple concave tabletop outline into convex triangular prisms."""
import xml.etree.ElementTree as ET
import numpy as np


def triangulate(points):
    points=np.asarray(points,dtype=float)
    cross=lambda a,b: float(a[0]*b[1]-a[1]*b[0])
    area=sum(cross(a,b) for a,b in zip(points,np.roll(points,-1,axis=0)))
    remaining=list(range(len(points)))
    if area<0:remaining.reverse()
    triangles=[]
    while len(remaining)>3:
        for k,b in enumerate(remaining):
            a,c=remaining[k-1],remaining[(k+1)%len(remaining)]
            pa,pb,pc=points[[a,b,c]]
            if cross(pb-pa,pc-pb)<=1e-10:continue
            def inside(p):
                return min(cross(pb-pa,p-pa),cross(pc-pb,p-pb),cross(pa-pc,p-pc))>=-1e-10
            if any(inside(points[j]) for j in remaining if j not in (a,b,c)):continue
            triangles.append((a,b,c));remaining.pop(k);break
        else:raise ValueError('Table outline must be a simple polygon without duplicate vertices')
    triangles.append(tuple(remaining))
    return points,triangles


def add_table(root,outline,thickness):
    points,triangles=triangulate(outline)
    asset=root.find('asset');world=root.find('worldbody')
    verts=[[x,y,z] for z in [-thickness-0.003,-0.003] for x,y in points]
    n=len(points);faces=[]
    for a,b,c in triangles:faces.extend([(c,b,a),(a+n,b+n,c+n)])
    for a in range(n):
        b=(a+1)%n;faces.extend([(a,b,b+n),(a,b+n,a+n)])
    def flat(xs):return ' '.join(str(v) for row in xs for v in row)
    ET.SubElement(asset,'mesh',name='table_visual_mesh',vertex=flat(verts),face=flat(faces))
    ET.SubElement(world,'geom',name='table',type='mesh',mesh='table_visual_mesh',rgba='0.91 0.9 0.85 1',contype='0',conaffinity='0')
    for i,tri in enumerate(triangles):
        vertices=[[float(points[j,0]),float(points[j,1]),z] for z in [-thickness-0.003,-0.003] for j in tri]
        name=f'table_collision_{i}'
        ET.SubElement(asset,'mesh',name=name,vertex=flat(vertices))
        ET.SubElement(world,'geom',name=name,type='mesh',mesh=name,group='3',rgba='0.91 0.9 0.85 1')

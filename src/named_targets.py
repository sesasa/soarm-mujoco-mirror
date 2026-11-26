"""Session labels for observed objects; no language model or device access."""
import re
from adaptive_reach import TargetWatch


def normalize_name(text):
    name=' '.join(text.lower().strip().split())
    if not re.fullmatch(r'[a-z0-9][a-z0-9 _-]{0,39}',name):
        raise ValueError('Use a short name: letters, numbers, spaces or hyphens')
    return name


class NamedTargets:
    def __init__(self):self.targets={}
    def label(self,name,observation,now):
        name=normalize_name(name)
        self.targets[name]=TargetWatch(observation,now)
        return name
    def update(self,observations,now):
        for watch in self.targets.values():watch.update(observations,now)
    def resolve(self,text,now):
        text=' '.join(text.lower().strip().rstrip('.!?').split())
        name=re.sub(r'^(?:please )?(?:reach for|reach|go to|move to) (?:the )?', '', text)
        name=normalize_name(name)
        if name not in self.targets:raise ValueError('Unknown name. Click its object, then N to label it.')
        watch=self.targets[name]
        if not watch.stable(now):raise ValueError(name+': missing, moving or ambiguous; separate objects or relabel.')
        return name,watch

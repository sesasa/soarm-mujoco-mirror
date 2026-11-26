"""Explicit typed execution intent; never inferred from GPT's response."""
import re


def parse_pick_place_command(text):
    text=' '.join(text.strip().split())
    match=re.match(r'^(?:please )?pick and place(?:\s+|$)',text,re.I)
    if not match:return False,text
    target=text[match.end():].strip().rstrip('.!?')
    target=re.sub(r'\s+(?:in|into|on) (?:the )?plate$','',target,flags=re.I)
    target=re.sub(r'^the(?:\s+|$)','',target,flags=re.I).strip()
    if not target:raise ValueError('Name an object, for example: pick and place the highlighter')
    return True,target

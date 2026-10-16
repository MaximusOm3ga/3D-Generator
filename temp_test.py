import objaverse

uid = "1d3687a764404fc9a2ff78a06b873ecd"

# 1. Is this UID actually in LVIS's "chair" list?
lvis = objaverse.load_lvis_annotations()
chair_uids = set(lvis["chair"])
print(f"'chair' category has {len(chair_uids)} objects")
print(f"Is this UID in it? {uid in chair_uids}")

# 2. If it IS in there, what does Objaverse's own (Sketchfab-sourced) name
#    say this object actually is? This tells us if it's an LVIS mislabel.
annotations = objaverse.load_annotations([uid])
info = annotations.get(uid, {})
print(f"Sketchfab name: {info.get('name')}")
print(f"Sketchfab tags: {[t.get('name') for t in info.get('tags', [])]}")
print(f"Sketchfab categories: {[c.get('name') for c in info.get('categories', [])]}")
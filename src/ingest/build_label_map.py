"""
Define the first-pass (medium-granularity) flat label map and report real image counts.

Maps all 257 source categories -> ~28 target classes, then counts SINGLE-annotation
images per target (multi-annotation images are dropped from the first pass).
"""

#%% Imports and constants

import json, os, csv
from collections import Counter, defaultdict

metadata_file = r"E:\data\california-small-animals\california_small_animals_with_sequences.json"
output_file = r"C:\temp\california-small-animals-output"


#%% Support functions

def low(x):
    if x is None: return ""
    if isinstance(x, float) and x != x: return ""
    return str(x).strip().lower()

def target_class(cat):

    n = low(cat["name"]); cl = low(cat.get("class")); o = low(cat.get("order"))
    fa = low(cat.get("family")); ge = low(cat.get("genus"))

    # --- empties / special ---
    if n in ("blank", "misfire"): return "blank"
    if n == "setup_pickup": return "setup_pickup"
    if n in ("unknown", "animal", "no cv result"): return "EXCLUDE"
    if n == "lizards and snakes": return "EXCLUDE"   # ambiguous lizard vs snake

    # --- invertebrates ---
    if cl == "insecta": return "insect"
    if cl == "malacostraca": return "isopod_crustacean"
    if cl == "arachnida": return "spider_arachnid"
    if cl in ("gastropoda", "diplopoda", "chilopoda"): return "other_invert"

    # --- amphibians (frogs/toads/salamanders) ---
    if cl == "amphibia": return "amphibian"

    # --- birds ---
    if cl == "aves": return "bird"

    # --- reptiles ---
    if cl == "reptilia":
        # horned lizards have order mis-set to 'phrynosomatidae' (dataset quirk)
        if fa == "phrynosomatidae" or o == "phrynosomatidae" or n == "sceloporus/uta species":
            return "spiny_lizard"
        if fa == "teiidae": return "whiptail"
        if fa == "anguidae": return "alligator_lizard"
        if fa == "scincidae": return "skink"
        if fa in ("viperidae", "viperdae"): return "rattlesnake"
        if fa in ("colubridae", "leptotyphlopidae", "boidae"): return "snake"
        if fa in ("iguanidae", "crotaphytidae", "gekkonidae"): return "lizard_other"
        if o == "testudines": return "EXCLUDE"            # turtles, n=26
        if o == "squamata": return "lizard_other"          # generic squamata w/ family unset
        return "EXCLUDE"                                    # generic 'reptile'

    # --- mammals ---
    if cl == "mammalia":
        if o == "rodentia":
            if fa == "sciuridae":
                if ge == "neotamias" or "chipmunk" in n: return "chipmunk"
                return "squirrel"
            if fa == "heteromyidae": return "kangaroo_rat_pocket_mouse"
            if fa == "geomyidae": return "pocket_gopher"
            if fa == "dipodidae": return "mouse"                 # jumping mouse
            # voles
            if ge == "microtus" or "vole" in n or "arvicolinae" in n: return "vole"
            # woodrats / rats
            if ge in ("neotoma", "rattus", "sigmodon") or "woodrat" in n \
               or n in ("house rat", "brown rat", "woodrat or rat species"): return "woodrat_rat"
            # mice
            if ge in ("peromyscus", "reithrodontomys", "onychomys", "mus") \
               or n == "mouse species" or ("mouse" in n and "pocket" not in n and "kangaroo" not in n):
                return "mouse"
            # ambiguous / generic rodents (incl. 'rodent', cricetidae/muridae family,
            # 'woodrat or rat or mouse species', porcupine, 'small mammal')
            return "rodent_other"
        if o == "lagomorpha": return "rabbit_hare"
        if o == "eulipotyphla": return "shrew_mole"             # shrews + moles
        if o == "didelphimorphia": return "opossum"
        if o == "carnivora":
            if fa == "mephitidae": return "skunk"
            if fa == "mustelidae": return "weasel"
            return "mammal_other"                               # canids/felids/ursids/procyonids
        if n == "small mammal": return "rodent_other"
        return "mammal_other"                                   # other orders + generic 'mammal'

    return "EXCLUDE"

# ...def target_class(...)

#%% Execution

print("Loading...", flush=True)
with open(metadata_file, encoding="utf-8") as f:
    data = json.load(f)
cats = {c["id"]: c for c in data["categories"]}
anns = data["annotations"]

# map source category -> target
src2tgt = {cid: target_class(c) for cid, c in cats.items()}

# annotation counts per source category
acount = Counter(a["category_id"] for a in anns)

# image-level: keep only single-annotation images
img_anncount = Counter(a["image_id"] for a in anns)
img_firstcat = {}
for a in anns:
    img_firstcat.setdefault(a["image_id"], a["category_id"])

tgt_img = Counter()         # single-annotation image count per target
multi_dropped = 0
for iid, n in img_anncount.items():
    if n > 1:
        multi_dropped += 1
        continue
    tgt = src2tgt[img_firstcat[iid]]
    tgt_img[tgt] += 1

# build per-target -> contributing sources
tgt_sources = defaultdict(list)
for cid, c in cats.items():
    tgt_sources[src2tgt[cid]].append((c["name"], acount.get(cid, 0)))

ORDER = ["blank","setup_pickup",
         "mouse","vole","woodrat_rat","kangaroo_rat_pocket_mouse","squirrel","chipmunk",
         "pocket_gopher","rodent_other","rabbit_hare","shrew_mole","skunk","weasel",
         "opossum","mammal_other",
         "spiny_lizard","whiptail","alligator_lizard","skink","snake","rattlesnake","lizard_other",
         "amphibian","bird","insect","isopod_crustacean","spider_arachnid","other_invert",
         "EXCLUDE"]

print(f"\nMulti-annotation images dropped: {multi_dropped}")
print(f"\n{'TARGET CLASS':<26}{'#images(single-ann)':>20}")
print("-"*46)
total_train = 0
for t in ORDER:
    if t == "EXCLUDE": continue
    print(f"{t:<26}{tgt_img.get(t,0):>20,}")
    total_train += tgt_img.get(t,0)
print("-"*46)
print(f"{'TOTAL (incl blank)':<26}{total_train:>20,}")
print(f"{'EXCLUDED images':<26}{tgt_img.get('EXCLUDE',0):>20,}")

print("\n\n===== SOURCE CATEGORIES PER TARGET (audit) =====")
for t in ORDER:
    srcs = sorted(tgt_sources.get(t, []), key=lambda x:-x[1])
    tot = sum(s[1] for s in srcs)
    print(f"\n### {t}  (ann total {tot:,}; {len(srcs)} source cats)")
    for name, n in srcs:
        print(f"   {n:>9,}  {name}")

# write csv
with open(os.path.join(output_file,"label_map.csv"),"w",newline="",encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["source_category","ann_count","target_class"])
    for cid,c in sorted(cats.items(), key=lambda kv:-acount.get(kv[0],0)):
        w.writerow([c["name"], acount.get(cid,0), src2tgt[cid]])
print("\nWrote label_map.csv")

"""

Canonical source-category -> first-pass target-class mapping (labels only; no paths).

Single source of truth for the label set used by the manifest/split/copy/train/eval pipeline. The
medium-granularity 29-class label set (27 animal classes + blank + setup_pickup) was agreed with
the dataset maintainer; see README.md and label_map_report.txt. Machine-specific paths
(METADATA_FILE, IMAGE_ROOT, OUTPUT_ROOT, TRAIN_ROOT, EXCLUDE_FILES) now live in a JSON path-config
loaded via
``path_config.load_path_config`` (see path_config.py), so this module is pure label logic and
imports cleanly on any machine.

"""

#%% Imports and constants

# Canonical class order (index = training label id). 'EXCLUDE' is not a class;
# images mapping to it are dropped from training.
CLASS_ORDER = [
    "blank", "setup_pickup",
    "mouse", "vole", "rodent_other", "woodrat_rat", "squirrel",
    "kangaroo_rat_pocket_mouse", "chipmunk", "pocket_gopher",
    "rabbit_hare", "shrew_mole", "skunk", "weasel", "opossum", "mammal_other",
    "spiny_lizard", "whiptail", "snake", "alligator_lizard", "skink",
    "rattlesnake", "lizard_other",
    "amphibian", "bird", "insect", "isopod_crustacean", "spider_arachnid",
    "other_invert",
]


#%% Support functions

def low(x):
    if x is None:
        return ""
    if isinstance(x, float) and x != x:  # NaN
        return ""
    return str(x).strip().lower()


#%% Category mapping functions

def target_class(cat):
    """
    Map a source category dict (name/class/order/family/genus) -> target class.

    Returns 'EXCLUDE' for images that should be dropped from training.
    """

    n = low(cat["name"]); cl = low(cat.get("class")); o = low(cat.get("order"))
    fa = low(cat.get("family")); ge = low(cat.get("genus"))

    # --- empties / special ---
    if n in ("blank", "misfire"):
        return "blank"
    if n == "setup_pickup":
        return "setup_pickup"
    if n in ("unknown", "animal", "no cv result"):
        return "EXCLUDE"
    if n == "lizards and snakes":
        return "EXCLUDE"  # ambiguous lizard vs snake

    # --- invertebrates ---
    if cl == "insecta":
        return "insect"
    if cl == "malacostraca":
        return "isopod_crustacean"
    if cl == "arachnida":
        return "spider_arachnid"
    if cl in ("gastropoda", "diplopoda", "chilopoda"):
        return "other_invert"

    # --- amphibians (frogs/toads/salamanders) ---
    if cl == "amphibia":
        return "amphibian"

    # --- birds ---
    if cl == "aves":
        return "bird"

    # --- reptiles ---
    if cl == "reptilia":
        # NOTE: in the *original* JSON, zebra-tailed lizard had order=phrynosomatidae;
        # the o== check below is harmless belt-and-suspenders for the fixed file.
        if fa == "phrynosomatidae" or o == "phrynosomatidae" or n == "sceloporus/uta species":
            return "spiny_lizard"
        if fa == "teiidae":
            return "whiptail"
        if fa == "anguidae":
            return "alligator_lizard"
        if fa == "scincidae":
            return "skink"
        if fa in ("viperidae", "viperdae"):
            return "rattlesnake"
        if fa in ("colubridae", "leptotyphlopidae", "boidae"):
            return "snake"
        if fa in ("iguanidae", "crotaphytidae", "gekkonidae"):
            return "lizard_other"
        if o == "testudines":
            return "EXCLUDE"          # turtles, n=26
        if o == "squamata":
            return "lizard_other"      # generic squamata, family unset
        return "EXCLUDE"               # generic 'reptile'

    # --- mammals ---
    if cl == "mammalia":
        if o == "rodentia":
            if fa == "sciuridae":
                if ge == "neotamias" or "chipmunk" in n:
                    return "chipmunk"
                return "squirrel"
            if fa == "heteromyidae":
                return "kangaroo_rat_pocket_mouse"
            if fa == "geomyidae":
                return "pocket_gopher"
            if fa == "dipodidae":
                return "mouse"  # jumping mouse
            if ge == "microtus" or "vole" in n or "arvicolinae" in n:
                return "vole"
            if (ge in ("neotoma", "rattus", "sigmodon") or "woodrat" in n
                    or n in ("house rat", "brown rat", "woodrat or rat species")):
                return "woodrat_rat"
            if (ge in ("peromyscus", "reithrodontomys", "onychomys", "mus")
                    or n == "mouse species"
                    or ("mouse" in n and "pocket" not in n and "kangaroo" not in n)):
                return "mouse"
            return "rodent_other"  # 'rodent', cricetidae/muridae family, 'woodrat or rat or mouse', porcupine
        if o == "lagomorpha":
            return "rabbit_hare"
        if o == "eulipotyphla":
            return "shrew_mole"
        if o == "didelphimorphia":
            return "opossum"
        if o == "carnivora":
            if fa == "mephitidae":
                return "skunk"
            if fa == "mustelidae":
                return "weasel"
            return "mammal_other"
        if n == "small mammal":
            return "rodent_other"
        return "mammal_other"

    return "EXCLUDE"

# def target_class(...)

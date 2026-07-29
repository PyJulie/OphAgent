"""Native category vocabularies of the CFP CLIP fleet, verbatim from each model's
official source (verified — see memory project_clip_native_vocab), plus the
curated clinical screening superset and the per-model query->native-label map.

WHY: the legacy adapters hard-truncated each model to a small fixed subset (e.g.
RetiZero -> 16 labels, FLAIR's ~109 -> canon-11), which dropped whole conditions
(retinal artery occlusion, hypertensive subtypes, AION, CSC subtypes, ...). All
three models are OPEN-VOCAB CLIP scorers, so we restore their native vocabularies
and, at executor time, map the task's target conditions onto each model's native
label(s) (allowing "no native equivalent"). RAO unlock verified locally: RetiZero
+ native RAO label -> 17/17 RAO+ in top-3.

Each model's `classify(model, image, categories, use_domain_knowledge=...)` takes
an arbitrary category list and softmax-scores it; FLAIR/RetiZero additionally
mean-pool expert descriptions when use_domain_knowledge=True.
"""
from __future__ import annotations

# ── FLAIR (jusiro/FLAIR) — `definitions` keys, verbatim. Open-vocab; native
#    CRAO + BRAO. Template "A fundus photograph of [CLS]" + domain-knowledge
#    ensembling over the expert descriptions in dictionary.py.
FLAIR_NATIVE = [
    "no diabetic retinopathy", "mild diabetic retinopathy", "moderate diabetic retinopathy",
    "severe diabetic retinopathy", "proliferative diabetic retinopathy", "diabetic retinopathy",
    "hard exudates", "soft exudates", "microaneurysms", "haemorrhages", "cotton wool spots",
    "age related macular degeneration", "neovascular age related macular degeneration",
    "geographical age related macular degeneration", "drusens", "media haze", "pathologic myopia",
    "myopic maculopathy grade cero", "myopic maculopathy grade one", "myopic maculopathy grade two",
    "myopic maculopathy grade three", "myopic maculopathy grade four",
    "branch retinal vein occlusion", "central retinal vein occlusion",
    "central retinal artery occlusion", "branch retinal artery occlusion",
    "tessellation", "epiretinal membrane", "laser scar", "macular scar", "central serous retinopathy",
    "acute central serous retinopathy", "chronic central serous retinopathy",
    "optic disc cupping", "optic disc pallor", "optic disc edema", "anterior ischemic optic neuropathy",
    "tortuous vessels", "asteroid hyalosis", "shunt", "parafoveal telangiectasia", "retinal traction",
    "retinitis", "chorioretinitis", "retinal pigment epithelium changes", "macular hole",
    "retinitis pigmentosa", "colobomas", "optic disc pit maculopathy", "preretinal haemorrhage",
    "myelinated nerve fibers", "haemorrhagic retinopathy", "tilted disc", "cystoid macular edema",
    "post traumatic choroidal rupture", "choroidal folds", "vitreous haemorrhage", "macroaneurysm",
    "vasculitis", "plaque", "haemorrhagic pigment epithelial detachment", "collaterals",
    "large optic cup", "retina detachment", "maculopathy", "glaucoma", "optic atrophy",
    "severe hypertensive retinopathy", "mild hypertensive retinopathy",
    "moderate hypertensive retinopathy", "malignant hypertensive retinopathy", "hypertensive retinopathy",
    "disc swelling and elevation", "dragged disk", "congenital disk abnormality",
    "Bietti crystalline dystrophy", "peripheral retinal degeneration and break", "neoplasm",
    "yellow-white spots flecks", "fibrosis", "cataract", "nevus", "normal",
]
FLAIR_TEMPLATE = "A fundus photograph of [CLS]"

# ── RetiZero (LooKing9218/RetiZero) — Zeroshot demo labels, verbatim (open-vocab).
RETIZERO_NATIVE = [
    "Normal", "Retinal Vein Occlusion", "Central Serous Chorioretinopathy",
    "Non-proliferative Diabetic Retinopathy", "Proliferative Diabetic Retinopathy",
    "Epiretinal Membrane", "Glaucoma", "Macular Hole", "Pathologic Maculopathy",
    "Retinal Artery Occlusion", "Retinal Detachment", "Retinitis Pigmentosa",
    "Vogt-Koyanagi-Harada (VKH) disease", "Age-related Macular Degeneration",
]
RETIZERO_TEMPLATE = "A fundus photograph of [CLS]"

# ── ViLReF (ViT-B/16, Chinese) — 33 native categories (Chinese), verbatim.
#    NO native artery-occlusion (only generic 血管阻塞). Generic CN-CLIP templates.
VILREF_NATIVE_ZH = [
    "正常眼底", "白内障", "动脉硬化", "糖尿病视网膜病变", "飞蚊症", "近视", "老视", "青光眼",
    "脉络膜视网膜病变", "出血", "交叉压迹", "豹纹眼底", "动脉细", "玻璃体后脱离", "血管阻塞",
    "硬渗", "黄斑变性", "大视杯", "玻璃膜疣", "萎缩弧", "新生血管", "微动脉瘤", "神经纤维层缺损",
    "视网膜脱离", "激光斑", "色素上皮层脱离", "脉络膜萎缩", "模糊眼底", "黄斑区色素紊乱", "棉絮斑",
]

# ── Curated clinical screening SUPERSET (canonical taxonomy). full-pathology
#    target = this set (~26), NOT all ~109 and NOT just the task 7. Narrowed at
#    runtime by vision candidate_directions + effort tier.
CANONICAL_SUPERSET = [
    "Normal", "Diabetic retinopathy", "Proliferative diabetic retinopathy",
    "Age-related macular degeneration", "Glaucoma", "Retinal vein occlusion",
    "Retinal artery occlusion", "Pathological myopia", "Epiretinal membrane",
    "Macular hole", "Retinal detachment", "Central serous chorioretinopathy",
    "Hypertensive retinopathy", "Cataract", "Optic disc pallor / atrophy",
    "Optic disc edema / papilledema", "Anterior ischemic optic neuropathy",
    "Retinitis pigmentosa", "Drusen", "Laser scar / treated retinopathy",
    "Macular scar", "Tessellated fundus", "Chorioretinitis / retinitis",
    "Vitreous haemorrhage", "Macroaneurysm", "Myelinated nerve fibers",
]

# The 7-condition multi-label task taxonomy (always included in any target set).
TASK7 = ["DR", "AMD", "Glaucoma", "RVO", "PM", "ERM", "RAO"]

# Effort -> how wide the broad screen target is (coverage/depth, not decision).
EFFORT_TARGET_SIZE = {"low": 8, "medium": 14, "high": 26, "max": 26, "ultra": 26}


# ── Static canonical -> per-model native-label map. This is exactly what the
#    EXECUTOR mapping layer caches for the fixed canonical superset (the LLM
#    generates the same for arbitrary ad-hoc queries). None = "no native
#    equivalent for this model" (do not force a bad match — e.g. ViLReF has no
#    artery-occlusion concept). A canonical class may map to SEVERAL native
#    labels (synonyms / subtypes); we take the max over them per model.
CANON_TO_NATIVE: dict[str, dict[str, list[str] | None]] = {
    "Normal": {"flair": ["normal"], "retizero": ["Normal"], "vilref": ["正常眼底"]},
    "Diabetic retinopathy": {"flair": ["diabetic retinopathy", "moderate diabetic retinopathy"],
                             "retizero": ["Non-proliferative Diabetic Retinopathy"], "vilref": ["糖尿病视网膜病变"]},
    "Proliferative diabetic retinopathy": {"flair": ["proliferative diabetic retinopathy"],
                             "retizero": ["Proliferative Diabetic Retinopathy"], "vilref": ["新生血管"]},
    "Age-related macular degeneration": {"flair": ["age related macular degeneration", "drusens"],
                             "retizero": ["Age-related Macular Degeneration"], "vilref": ["黄斑变性", "玻璃膜疣"]},
    "Glaucoma": {"flair": ["glaucoma", "optic disc cupping"], "retizero": ["Glaucoma"], "vilref": ["青光眼", "大视杯"]},
    "Retinal vein occlusion": {"flair": ["central retinal vein occlusion", "branch retinal vein occlusion"],
                             "retizero": ["Retinal Vein Occlusion"], "vilref": ["血管阻塞"]},
    "Retinal artery occlusion": {"flair": ["central retinal artery occlusion", "branch retinal artery occlusion"],
                             "retizero": ["Retinal Artery Occlusion"], "vilref": None},  # ViLReF: no native RAO
    "Pathological myopia": {"flair": ["pathologic myopia", "myopic maculopathy grade two"],
                             "retizero": ["Pathologic Maculopathy"], "vilref": ["脉络膜萎缩", "萎缩弧"]},
    "Epiretinal membrane": {"flair": ["epiretinal membrane"], "retizero": ["Epiretinal Membrane"], "vilref": None},
    "Macular hole": {"flair": ["macular hole"], "retizero": ["Macular Hole"], "vilref": None},
    "Retinal detachment": {"flair": ["retina detachment"], "retizero": ["Retinal Detachment"], "vilref": ["视网膜脱离"]},
    "Central serous chorioretinopathy": {"flair": ["central serous retinopathy"],
                             "retizero": ["Central Serous Chorioretinopathy"], "vilref": ["脉络膜视网膜病变"]},
    "Hypertensive retinopathy": {"flair": ["hypertensive retinopathy", "severe hypertensive retinopathy"],
                             "retizero": None, "vilref": ["动脉硬化", "交叉压迹"]},
    "Cataract": {"flair": ["cataract"], "retizero": None, "vilref": ["白内障"]},
    "Optic disc pallor / atrophy": {"flair": ["optic disc pallor", "optic atrophy"], "retizero": None, "vilref": None},
    "Optic disc edema / papilledema": {"flair": ["optic disc edema", "disc swelling and elevation"], "retizero": None, "vilref": None},
    "Anterior ischemic optic neuropathy": {"flair": ["anterior ischemic optic neuropathy"], "retizero": None, "vilref": None},
    "Retinitis pigmentosa": {"flair": ["retinitis pigmentosa"], "retizero": ["Retinitis Pigmentosa"], "vilref": None},
    "Drusen": {"flair": ["drusens"], "retizero": None, "vilref": ["玻璃膜疣"]},
    "Laser scar / treated retinopathy": {"flair": ["laser scar"], "retizero": None, "vilref": ["激光斑"]},
    "Macular scar": {"flair": ["macular scar"], "retizero": None, "vilref": None},
    "Tessellated fundus": {"flair": ["tessellation"], "retizero": None, "vilref": ["豹纹眼底"]},
    "Chorioretinitis / retinitis": {"flair": ["chorioretinitis", "retinitis"], "retizero": None, "vilref": None},
    "Vitreous haemorrhage": {"flair": ["vitreous haemorrhage"], "retizero": None, "vilref": ["出血"]},
    "Macroaneurysm": {"flair": ["macroaneurysm"], "retizero": None, "vilref": ["微动脉瘤"]},
    "Myelinated nerve fibers": {"flair": ["myelinated nerve fibers"], "retizero": None, "vilref": None},
}

# Map the 7-condition task taxonomy to canonical names (for scoring the task eval).
TASK7_TO_CANON = {
    "DR": "Diabetic retinopathy", "AMD": "Age-related macular degeneration",
    "Glaucoma": "Glaucoma", "RVO": "Retinal vein occlusion",
    "PM": "Pathological myopia", "ERM": "Epiretinal membrane",
    "RAO": "Retinal artery occlusion",
}

"""
Registry of all publicly available OCT and OCTA datasets.

Each entry contains metadata, download source, and instructions.
Sources: Kaggle, Mendeley Data, Zenodo, institutional repositories.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class TaskType(Enum):
    CLASSIFICATION = "classification"
    SEGMENTATION = "segmentation"
    DETECTION = "detection"
    GENERATION = "generation"
    PREDICTION = "prediction"


class Modality(Enum):
    OCT_2D = "oct_2d"
    OCT_3D = "oct_3d"
    OCTA = "octa"
    FUNDUS = "fundus"
    MULTIMODAL = "multimodal"


class DownloadSource(Enum):
    KAGGLE = "kaggle"
    MENDELEY = "mendeley"
    ZENODO = "zenodo"
    GDRIVE = "gdrive"
    DIRECT_URL = "direct_url"
    GITHUB = "github"
    INSTITUTIONAL = "institutional"


@dataclass
class DatasetInfo:
    name: str
    description: str
    size: str
    modality: Modality
    tasks: list[TaskType]
    diseases: list[str]
    annotations: list[str]
    source: DownloadSource
    source_id: str
    url: str
    citation: str
    num_classes: int | None = None
    class_names: list[str] = field(default_factory=list)
    notes: str = ""


DATASET_REGISTRY: dict[str, DatasetInfo] = {
    # ═══════════════════════════════════════════════════════════════════════
    # CLASSIFICATION DATASETS
    # ═══════════════════════════════════════════════════════════════════════
    "kermany": DatasetInfo(
        name="Kermany OCT",
        description="Largest public OCT dataset. 207K B-scans across 4 categories. "
        "Tiered expert grading, single-device (Spectralis).",
        size="207,130 B-scans",
        modality=Modality.OCT_2D,
        tasks=[TaskType.CLASSIFICATION],
        diseases=["CNV", "DME", "Drusen", "Normal"],
        annotations=["Class Labels"],
        source=DownloadSource.KAGGLE,
        source_id="paultimothymooney/kermany2018",
        url="https://www.kaggle.com/datasets/paultimothymooney/kermany2018",
        citation="Kermany et al. 2018",
        num_classes=4,
        class_names=["CNV", "DME", "DRUSEN", "NORMAL"],
    ),
    "octdl": DatasetInfo(
        name="OCTDL",
        description="Broader disease spectrum than Kermany. "
        "Acquired with Optovue Avanti RTVue XR.",
        size="2,064 B-scans",
        modality=Modality.OCT_2D,
        tasks=[TaskType.CLASSIFICATION],
        diseases=["AMD", "DME", "ERM", "RAO", "RVO", "VID", "Normal"],
        annotations=["Class Labels"],
        source=DownloadSource.KAGGLE,
        source_id="orvile/octdl-optical-coherence-tomography-dataset",
        url="https://www.kaggle.com/datasets/orvile/octdl-optical-coherence-tomography-dataset",
        citation="OCTDL, Kulyabin et al.",
        num_classes=7,
        class_names=["AMD", "DME", "ERM", "RAO", "RVO", "VID", "NORMAL"],
    ),
    "oct_c8": DatasetInfo(
        name="OCT-C8",
        description="Eight-class dataset with pre-divided train/val/test splits.",
        size="24,000 B-scans",
        modality=Modality.OCT_2D,
        tasks=[TaskType.CLASSIFICATION],
        diseases=["AMD", "CNV", "CSR", "DME", "DR", "Drusen", "MH", "Normal"],
        annotations=["Class Labels"],
        source=DownloadSource.KAGGLE,
        source_id="obulisainaren/retinal-oct-c8",
        url="https://www.kaggle.com/datasets/obulisainaren/retinal-oct-c8",
        citation="OCT-C8",
        num_classes=8,
        class_names=["AMD", "CNV", "CSR", "DME", "DR", "DRUSEN", "MH", "NORMAL"],
    ),
    "srinivasan": DatasetInfo(
        name="Srinivasan OCT",
        description="Early benchmark dataset. 45 volumes across 3 categories.",
        size="45 volumes (3 categories)",
        modality=Modality.OCT_3D,
        tasks=[TaskType.CLASSIFICATION],
        diseases=["AMD", "DME", "Normal"],
        annotations=["Class Labels"],
        source=DownloadSource.INSTITUTIONAL,
        source_id="",
        url="https://people.duke.edu/~sf59/Srinivasan_BOE_2014_dataset.htm",
        citation="Srinivasan et al. BOE 2014",
        num_classes=3,
        class_names=["AMD", "DME", "NORMAL"],
    ),

    # ═══════════════════════════════════════════════════════════════════════
    # SEGMENTATION DATASETS
    # ═══════════════════════════════════════════════════════════════════════
    "duke_dme": DatasetInfo(
        name="Duke DME",
        description="110 B-scans from 10 subjects with 8 retinal layer boundaries. "
        "Acquired with Heidelberg Spectralis.",
        size="110 B-scans (10 subjects)",
        modality=Modality.OCT_2D,
        tasks=[TaskType.SEGMENTATION],
        diseases=["DME"],
        annotations=["Retinal Layer boundaries (8)"],
        source=DownloadSource.INSTITUTIONAL,
        source_id="",
        url="https://people.duke.edu/~sf59/Chiu_BOE_2015_dataset.htm",
        citation="Chiu et al. BOE 2015",
    ),
    "retouch": DatasetInfo(
        name="RETOUCH",
        description="Multi-vendor challenge dataset for fluid segmentation. "
        "3 OCT devices (Cirrus, Triton, Spectralis). 3D volumes.",
        size="70 volumes",
        modality=Modality.OCT_3D,
        tasks=[TaskType.SEGMENTATION],
        diseases=["AMD", "RVO"],
        annotations=["Fluid Labels (IRF, SRF, PED)"],
        source=DownloadSource.INSTITUTIONAL,
        source_id="",
        url="https://retouch.grand-challenge.org/",
        citation="RETOUCH Challenge, Bogunovic et al. 2019",
        notes="Requires challenge registration. Multi-device for domain robustness.",
    ),
    "aroi": DatasetInfo(
        name="AROI",
        description="Expert pixel-wise annotations for layers and fluid in nAMD. "
        "Acquired with Zeiss Cirrus HD OCT 4000.",
        size="1,136 B-scans (24 subjects)",
        modality=Modality.OCT_2D,
        tasks=[TaskType.SEGMENTATION],
        diseases=["AMD (nAMD)"],
        annotations=["Retinal Layers", "Fluid Labels"],
        source=DownloadSource.ZENODO,
        source_id="",
        url="https://ipg.fer.hr/ipg/resources/oct_image_database",
        citation="Melinscak et al.",
    ),
    "oimhs": DatasetInfo(
        name="OIMHS",
        description="Large single-disease segmentation dataset for macular holes. "
        "Four segmentation labels.",
        size="3,859 B-scans (119 subjects)",
        modality=Modality.OCT_2D,
        tasks=[TaskType.SEGMENTATION],
        diseases=["Macular Hole"],
        annotations=["Seg (4 labels: retina, MH, intraretinal cysts, choroid)"],
        source=DownloadSource.GITHUB,
        source_id="",
        url="https://github.com/OIMHS/OIMHS-Dataset",
        citation="OIMHS",
    ),
    "oct5k": DatasetInfo(
        name="OCT5k",
        description="Largest multi-disease layer segmentation dataset. "
        "Multi-grader annotations with biomarker bounding boxes.",
        size="1,672 scans (5,016 labels)",
        modality=Modality.OCT_3D,
        tasks=[TaskType.SEGMENTATION, TaskType.DETECTION],
        diseases=["AMD", "DME", "Normal"],
        annotations=["Retinal Layer boundaries (5)", "BBox (9 classes)"],
        source=DownloadSource.KAGGLE,
        source_id="",
        url="https://oct5k.github.io/",
        citation="OCT5k, 2024",
        notes="Multi-grader annotations for studying inter-grader variability.",
    ),

    # ═══════════════════════════════════════════════════════════════════════
    # MULTIMODAL DATASETS
    # ═══════════════════════════════════════════════════════════════════════
    "gamma": DatasetInfo(
        name="GAMMA",
        description="Paired fundus photographs and OCT volumes for glaucoma detection.",
        size="300 samples",
        modality=Modality.MULTIMODAL,
        tasks=[TaskType.CLASSIFICATION, TaskType.SEGMENTATION],
        diseases=["Glaucoma"],
        annotations=["Class Labels", "Retinal Layers"],
        source=DownloadSource.INSTITUTIONAL,
        source_id="",
        url="https://aistudio.baidu.com/competition/detail/90/0/introduction",
        citation="GAMMA Challenge",
    ),
    "multieye": DatasetInfo(
        name="MultiEYE",
        description="Largest public multimodal OCT/fundus dataset. "
        "Unpaired multimodal design. Patient-wise splits.",
        size="58,036 fundus + 45,923 OCT B-scans",
        modality=Modality.MULTIMODAL,
        tasks=[TaskType.CLASSIFICATION],
        diseases=["AMD", "DR", "Glaucoma", "Myopia", "MEM", "CSC", "and others"],
        annotations=["Class Labels"],
        source=DownloadSource.KAGGLE,
        source_id="",
        url="https://huggingface.co/datasets/MultiEYE/MultiEYE",
        citation="MultiEYE, 2024",
        num_classes=9,
        notes="HuggingFace hosted. Multi-class.",
    ),
    "fundoct": DatasetInfo(
        name="FUND-OCT",
        description="Paired fundus and OCT images. Small scale.",
        size="105 subjects",
        modality=Modality.MULTIMODAL,
        tasks=[TaskType.CLASSIFICATION],
        diseases=["4 disease types"],
        annotations=["Class Labels"],
        source=DownloadSource.INSTITUTIONAL,
        source_id="",
        url="",
        citation="FUND-OCT",
    ),

    # ═══════════════════════════════════════════════════════════════════════
    # OCTA DATASETS
    # ═══════════════════════════════════════════════════════════════════════
    "octa500": DatasetInfo(
        name="OCTA-500",
        description="Largest public OCTA resource (>80 GB). "
        "Rich vessel annotations including artery/vein/capillary/FAZ.",
        size="500 subjects (361,600 scans)",
        modality=Modality.OCTA,
        tasks=[TaskType.SEGMENTATION, TaskType.CLASSIFICATION],
        diseases=["DR", "AMD", "Glaucoma", "CNV", "Normal"],
        annotations=["Retinal Layers", "Vessel (7 label types)", "Class Labels"],
        source=DownloadSource.INSTITUTIONAL,
        source_id="",
        url="https://ieee-dataport.org/open-access/octa-500",
        citation="OCTA-500, Li et al.",
        notes="Two fields of view (3mm/6mm). OCT and OCTA volumes.",
    ),
    "rose": DatasetInfo(
        name="ROSE",
        description="First public OCTA vessel segmentation dataset. "
        "En face angiograms with fine vessel annotations.",
        size="229 images (117 ROSE-1)",
        modality=Modality.OCTA,
        tasks=[TaskType.SEGMENTATION],
        diseases=["Microvascular structure"],
        annotations=["Seg (pixel + centerline)"],
        source=DownloadSource.GITHUB,
        source_id="",
        url="https://imed.nimte.ac.cn/dataofrose.html",
        citation="ROSE",
    ),

    # ═══════════════════════════════════════════════════════════════════════
    # SYNTHETIC / SPECIAL-PURPOSE
    # ═══════════════════════════════════════════════════════════════════════
    "syn_oct": DatasetInfo(
        name="SYN-OCT",
        description="Synthetic OCT dataset with perfect annotations. "
        "Useful for pretraining and data augmentation.",
        size="200,000 synthetic images",
        modality=Modality.OCT_2D,
        tasks=[TaskType.CLASSIFICATION, TaskType.SEGMENTATION, TaskType.GENERATION],
        diseases=["Glaucoma", "Healthy"],
        annotations=["Retinal Layers", "Seg"],
        source=DownloadSource.INSTITUTIONAL,
        source_id="",
        url="",
        citation="SYN-OCT",
        notes="Domain gap with real data limits direct clinical use.",
    ),
    "rasta": DatasetInfo(
        name="RASTA",
        description="SS-OCT volumes + en face + clinical data. "
        "Designed for systemic risk prediction (oculomics).",
        size="499 patients, 814 volumes, 2005 en face",
        modality=Modality.MULTIMODAL,
        tasks=[TaskType.CLASSIFICATION, TaskType.PREDICTION],
        diseases=["Cardiovascular risk"],
        annotations=["Class Labels (weak, clinical risk scores)"],
        source=DownloadSource.INSTITUTIONAL,
        source_id="",
        url="",
        citation="RASTA",
    ),
    "octid": DatasetInfo(
        name="OCTID",
        description="Acquired with Cirrus HD-OCT. "
        "Includes 25 manually segmented normal images.",
        size=">500 B-scans",
        modality=Modality.OCT_2D,
        tasks=[TaskType.CLASSIFICATION, TaskType.SEGMENTATION],
        diseases=["AMD", "CSR", "MH", "DR", "Normal"],
        annotations=["Class Labels", "Retinal Layers (25 images)"],
        source=DownloadSource.MENDELEY,
        source_id="",
        url="https://data.mendeley.com/datasets/rscbjbr9sj/3",
        citation="OCTID, Gholami et al.",
    ),
}


def list_datasets(
    task: TaskType | None = None,
    modality: Modality | None = None,
) -> list[DatasetInfo]:
    results = []
    for info in DATASET_REGISTRY.values():
        if task and task not in info.tasks:
            continue
        if modality and info.modality != modality:
            continue
        results.append(info)
    return results


def get_dataset(name: str) -> DatasetInfo:
    if name not in DATASET_REGISTRY:
        available = ", ".join(DATASET_REGISTRY.keys())
        raise KeyError(f"Unknown dataset '{name}'. Available: {available}")
    return DATASET_REGISTRY[name]

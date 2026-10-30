MODELS = {
    "ABMIL": ("models.ABMIL", "ABMIL"),
}

DATASETS = {
    "TCGA": {"slide_id_key": "slide_id"},
    "PANDA": {"slide_id_key": "image_id"},
    "CAMELYON16": {"slide_id_key": "fileId"},
    "IMP-CRS-2024": {"slide_id_key": "slide_name"},
}
DATASETS_ORGAN = {
    "TCGA-BLCA": "Bladder",
    "TCGA-BRCA": "Breast",
    "TCGA-CESC": "Cervix",
    "TCGA-GBM": "Brain",
    "TCGA-lung": "Lung",
    "TCGA-UCEC": "Uterus",
    "TCGA-UCS": "Uterus",
    "TCGA-UVM": "Uvea",
    "PANDA": "Prostate",
    "CAMELYON16": "Lymph Node",
    "IMP-CRS-2024": "Colorectal",
}

EMBEDDING_MODELS = {
    "GigaPath": {
        "dim": 1536,
        "folder_name": "tiles-embeddings",
    },
    "CONCH": {
        "dim": 512,
        "folder_name": "tiles-embeddings_conch",
    },
}

from pathlib import Path

# This is the directory where all the original SEM images are
original_path = Path("/mnt/d/cloud/OneDrive - Mass General Brigham/projects/drug_delivery/drug_delivery_sem_rami/data/1_cropped")
data_path = Path("/mnt/d/cloud/OneDrive - Mass General Brigham/projects/drug_delivery/drug_delivery_sem_rami/data")


original_images = list(map(str, list(original_path.rglob("*"))))
original_images.sort()
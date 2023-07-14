import sys
from pathlib import Path

if sys.platform == "darwin":
    data_path = Path("/Users/amir/Library/CloudStorage/OneDrive-MassGeneralBrigham/projects/drug_delivery"
                "/drug_delivery_sem_rami/data")

sample_original_path = Path("/Users/amir/Library/CloudStorage/OneDrive-MassGeneralBrigham/projects/drug_delivery"
                            "/drug_delivery_sem_rami/data/0_original/16-6754_46579_RSFA/16-6754_46579_RSFA_50x")

sample_splitted_images_path = Path("/Users/amir/Library/CloudStorage/OneDrive-MassGeneralBrigham/projects/drug_delivery"
                            "/drug_delivery_sem_rami/data/splitted/16-6754_46579_RSFA_50x0052")

sample_labels_path = Path("/Users/amir/Library/CloudStorage/OneDrive-MassGeneralBrigham/projects/drug_delivery"
                            "/drug_delivery_sem_rami/data/sample_labels/OneDrive_2_7-10-2023/")

training_data_path = Path("/Users/amir/Library/CloudStorage/OneDrive-MassGeneralBrigham/projects/drug_delivery"
                            "/drug_delivery_sem_rami/data/training_data/")

images_path = training_data_path / "images"
labels_path = training_data_path / "segmentations"

images = list(map(str, images_path.rglob("*.tif")))
images.sort()

labels = list(map(str, labels_path.rglob("*.tif")))
labels.sort()

# original_images = list(map(str, sample_original_path.rglob("*.tif")))
# original_images.sort()
#
# sample_splitted_images = list(map(str, sample_splitted_images_path.rglob("*.tif")))
# sample_splitted_images.sort()
#
# sample_labels = list(map(str, sample_labels_path.rglob("*.tif")))
# sample_labels.sort()


import os
import shutil


class DataIngestion:

    def __init__(self):

        self.source_path = "dataset"

        self.destination_path = os.path.join(
            "artifacts",
            "raw"
        )

    def initiate_data_ingestion(self):

        os.makedirs(
            self.destination_path,
            exist_ok=True
        )

        shutil.copytree(
            self.source_path,
            self.destination_path,
            dirs_exist_ok=True
        )

        print(
            f"Dataset copied successfully to {self.destination_path}"
        )

        return self.destination_path
import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt


class GTSRBLoader:
    def __init__(self):
        
        self.base_dir = os.path.dirname(os.path.dirname(__file__))

        
        self.dataset_path = os.path.join(self.base_dir, "dataset", "GTSRB")
        self.train_csv = os.path.join(self.dataset_path, "Train.csv")

        
        self.train_df = pd.read_csv(self.train_csv)     # load CSV

    def show_dataset_info(self):
        print("=" * 50)
        print("GTSRB DATASET INFORMATION")
        print("=" * 50)
        print(f"Total training images: {len(self.train_df)}")
        print(f"Number of classes: {self.train_df['ClassId'].nunique()}")
        print()

    def show_random_image(self, index=0):
        row = self.train_df.iloc[index]

        image_path = os.path.join(self.dataset_path, row["Path"])

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(4,4))
        plt.imshow(image)
        plt.title(f"Class ID: {row['ClassId']}")
        plt.axis("off")
        plt.show()


if __name__ == "__main__":
    loader = GTSRBLoader()

    loader.show_dataset_info()

    loader.show_random_image(0)
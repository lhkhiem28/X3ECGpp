
import os, sys
from libs import *

class ECGDataset(torch.utils.data.Dataset):
    def __init__(self, 
        df_path, data_path, 
        config, 
        augment = False, 
    ):
        self.df_path, self.data_path,  = df_path, data_path, 
        self.df = pandas.read_csv(self.df_path)

        self.config = config
        self.augment = augment

    def __len__(self, 
    ):
        return len(self.df)

    def drop_lead(self, 
        ecg, 
    ):
        if random.random() >= 0.5:
            ecg[np.random.randint(len(self.config["ecg_leads"])), :] = 0.0
        return ecg

    def __getitem__(self, 
        index, 
    ):
        row = self.df.iloc[index]

        ecg, demogr = np.load("{}/{}.npy".format(self.data_path, row["id"]))[self.config["ecg_leads"], :], [1 if grp == row["age"] else 0 for grp in range(8)] + [1 if grp == row["sex"] else 0 for grp in range(3)]
        ecg = pad_sequences(ecg, self.config["ecg_length"], "float64", 
            "post", "post", 
        )
        if self.augment:
            ecg = self.drop_lead(ecg)
        ecg, demogr = torch.tensor(ecg).float(), torch.tensor(demogr).float()

        if not self.config["is_multilabel"]:
            label = row["label"]
        else:
            label = row[[col for col in list(row.index) if "label_" in col]].values.astype("float64")

        return (ecg, demogr), row["r_count"], label
r""" Dataloader builder """
from torch.utils.data import DataLoader
from Datasetloader.XCAD_liot import DatasetXCAD_aug
from Datasetloader.DRIVE_LIOT import DatasetDRIVE_aug
from Datasetloader.STARE_LIOT import DatasetSTARE_aug
from Datasetloader.Cracktree import DatasetCrack_aug


class CSDataset:

    @classmethod
    def initialize(cls, datapath):

        cls.datasets = {
            'XCAD_LIOT': DatasetXCAD_aug,
            'DRIVE_LIOT':DatasetDRIVE_aug,
            'STARE_LIOT': DatasetSTARE_aug,
            'Cracktree_LIOT':DatasetCrack_aug
        }

        cls.datapath = datapath


    @classmethod
    def build_dataloader(cls, benchmark, bsz, nworker, split, img_mode, img_size, supervised):
        # Force randomness during training for diverse episode combinations
        # Freeze randomness during testing for reproducibility
        shuffle = split == 'train'
        nworker = nworker #if split == 'trn' else 0

        if split == 'train':
            dataset = cls.datasets[benchmark](benchmark,
                                              datapath=cls.datapath,
                                              split=split,
                                              img_mode=img_mode,
                                              img_size=img_size,
                                              supervised=supervised)
        else:
            dataset = cls.datasets[benchmark](benchmark,
                                              datapath=cls.datapath,
                                              split=split,
                                              img_mode='same',
                                              img_size=None,
                                              supervised=supervised)

        dataloader = DataLoader(dataset, batch_size=bsz, shuffle=shuffle, num_workers=nworker)

        return dataloader

import torch
import torchvision
from .QCDataset import QCDataset
import glob


class initialize_dataset:
    def __init__(self, image_resolution=320, batch_size=128, MNIST=True):
        self.image_resolution= image_resolution
        self.batch_size=batch_size
        self.MNIST=MNIST
        self.trainDir = '/Users/ravinderkaur/Documents/PathologyPipeline/QCTraining/TrainingImages/*'
        self.testDir = '/Users/ravinderkaur/Documents/PathologyPipeline/QCTraining/EvalImages/*'
        self.labelDir = '/Users/ravinderkaur/Documents/PathologyPipeline/QCTraining/SlideData/'
    
    def load_dataset(self, transform=False):
        trainList = glob.glob(self.trainDir)
        testList = glob.glob(self.testDir)

        train_dataset = QCDataset(trainList, self.labelDir)
        test_dataset = QCDataset(testList, self.labelDir)

        train_dataloader = torch.utils.data.DataLoader(dataset = train_dataset,
                                                        batch_size=self.batch_size,
                                                        shuffle=True)
        test_dataloader = torch.utils.data.DataLoader(dataset = test_dataset,
                                                        batch_size=self.batch_size,
                                                        shuffle=True)

        return train_dataloader, test_dataloader



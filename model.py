import pytorch_lightning
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.losses import DiceLoss
from monai.transforms import AsDiscrete,EnsureChannelFirstd,Compose,CropForegroundd,LoadImaged,Orientationd,RandCropByPosNegLabeld,ScaleIntensityRanged,Spacingd,EnsureType
from monai.metrics import DiceMetric
import os
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, list_data_collate, decollate_batch, DataLoader, Dataset
import torch 
from monai.utils import set_determinism
import glob
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

root_dir = "/mnt/Enterprise2/shirshak/"
data_dir = os.path.join(root_dir, "Task09_Spleen")

# set up loggers and checkpoints
log_dir = os.path.join("/home/shirshak/3D-SPLEEN-MONAI-Pt-Lightning", "logs")
tb_logger = pytorch_lightning.loggers.TensorBoardLogger(save_dir=log_dir)

train_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.5, 1.5, 2.0),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRanged(
                    keys=["image"],
                    a_min=-57,
                    a_max=164,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                ),
                CropForegroundd(keys=["image", "label"], source_key="image"),
                # randomly crop out patch samples from
                # big image based on pos / neg ratio
                # the image centers of negative samples
                # must be in valid image area
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=(96, 96, 96),
                    pos=1,
                    neg=1,
                    num_samples=4,
                    image_key="image",
                    image_threshold=0,
                ),
                # user can also add other random transforms
                #                 RandAffined(
                #                     keys=['image', 'label'],
                #                     mode=('bilinear', 'nearest'),
                #                     prob=1.0,
                #                     spatial_size=(96, 96, 96),
                #                     rotate_range=(0, 0, np.pi/15),
                #                     scale_range=(0.1, 0.1, 0.1)),
            ]
        )

val_transforms = Compose(
[
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.5, 1.5, 2.0),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRanged(
                    keys=["image"],
                    a_min=-57,
                    a_max=164,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                ),
                CropForegroundd(keys=["image", "label"], source_key="image"),
            ]
)

class SpleenDataset(Dataset):
    def __init__(self, file_names, transform):
        self.file_names = file_names
        self.transform = transform

    def __getitem__(self, index):
        file_names = self.file_names[index]
        dataset = self.transform(file_names) 
        return dataset
    
    def __len__(self):
        return len(self.file_names)



class Lightning_Model(pytorch_lightning.LightningModule):
    def __init__(self, max_epochs):
        super().__init__()
        self._model = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm=Norm.BATCH,
        )
        self.loss_function = DiceLoss(to_onehot_y=True, softmax=True)
        self.post_pred = Compose([EnsureType("tensor", device="cpu"), AsDiscrete(argmax=True, to_onehot=2)])
        self.post_label = Compose([EnsureType("tensor", device="cpu"), AsDiscrete(to_onehot=2)])
        self.dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
        self.best_val_dice = 0
        self.best_val_epoch = 0

        self.train_step_outputs = []
        self.validation_step_outputs = []
        self.max_epochs = max_epochs
        self.image_logger_tensorboard = []

    def forward(self, x):
        return self._model(x)

    def prepare_data(self):
        train_images = sorted(glob.glob(os.path.join(data_dir, "imagesTr", "*.nii.gz")))
        train_labels = sorted(glob.glob(os.path.join(data_dir, "labelsTr", "*.nii.gz")))
        data_dicts = [
            {"image": image_name, "label": label_name} for image_name, label_name in zip(train_images, train_labels)
        ]
        train_files, val_files = data_dicts[:-9], data_dicts[-9:]

        # set deterministic training for reproducibility
        set_determinism(seed=0)

        # self.train_ds = CacheDataset(data=train_files,transform=train_transforms,cache_rate=1.0,num_workers=4)
        # self.val_ds = CacheDataset(data=val_files,transform=val_transforms,cache_rate=1.0,num_workers=4)
        self.train_ds = SpleenDataset(file_names=train_files, transform=train_transforms)
        self.val_ds = SpleenDataset(file_names=val_files, transform=val_transforms)

    def train_dataloader(self):
        train_loader = DataLoader(
            self.train_ds,
            batch_size=2,
            shuffle=True,
            num_workers=4,
            collate_fn=list_data_collate,
        )
        return train_loader
    
    def val_dataloader(self):
        val_loader = DataLoader(self.val_ds, batch_size=1, num_workers=4)
        return val_loader
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self._model.parameters(), 1e-4)
        return optimizer
    
    def training_step(self,batch,batch_idx):
        images, labels = batch["image"], batch["label"]
        output = self.forward(images)
        loss = self.loss_function(output, labels)
        d = {"loss": loss, "train_number": len(output)}
        self.train_step_outputs.append(d)
        return d
    
    def on_train_epoch_end(self):
        train_loss, num_items = 0,0
        for output in self.train_step_outputs:
            train_loss += output["loss"].sum().item()
            num_items += output["train_number"]

        mean_train_loss = torch.tensor(train_loss/num_items)
        tensorboard_logs = {"train_loss":mean_train_loss}

        # Adding logs to TensorBoard
        tb_logger.experiment.add_scalar("Train Loss",mean_train_loss,self.current_epoch)

        self.train_step_outputs.clear()
        return {"log":tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        roi_size = (160, 160, 160)
        sw_batch_size = 4
        outputs = sliding_window_inference(images, roi_size, sw_batch_size, self.forward)
        loss = self.loss_function(outputs, labels)
        outputs = [self.post_pred(i) for i in decollate_batch(outputs)]
        labels = [self.post_label(i) for i in decollate_batch(labels)]
        self.dice_metric(y_pred=outputs, y=labels)
        dict = {"val_loss": loss, "val_number": len(outputs)}
        self.validation_step_outputs.append(dict)

        img_dict = {"image":images, "label":labels,"output":outputs}
        self.image_logger_tensorboard.append(img_dict)
        return dict
    
    def on_validation_epoch_end(self):
        val_loss, num_items = 0, 0
        for output in self.validation_step_outputs:
            val_loss += output["val_loss"].sum().item()
            num_items += output["val_number"]
        mean_val_dice = self.dice_metric.aggregate().item()
        self.dice_metric.reset()
        mean_val_loss = torch.tensor(val_loss / num_items)
        tensorboard_logs = {"val_dice": mean_val_dice,"val_loss": mean_val_loss}

        for count, logged_images in enumerate(self.image_logger_tensorboard):
            if count % 5 ==0:
                image = logged_images["image"]
                label = logged_images['label']
                out = logged_images['output']

                # print(np.stack((image),axis=0).shape) # (1, 1, 226, 157, 113)
                # print(np.stack((label),axis=0).shape) # (1, 2, 226, 157, 113)
                # print(np.stack((out),axis=0).shape) #(1, 2, 226, 157, 113) 

                image = torch.as_tensor(np.stack((image),axis=0)).squeeze(dim=0).squeeze(dim=0)
                label = torch.as_tensor(np.stack((label),axis=0)).squeeze(dim=0)
                out = torch.as_tensor(np.stack((out),axis=0)).squeeze(dim=0)

                fig, ax = plt.subplots(1,3)
                ax[0].set_title(f"image {count}")
                ax[0].imshow(image[:, :, 80], cmap="gray")
                
                ax[1].set_title(f"label {count}")
                ax[1].imshow(label.argmax(0)[:, :, 80])
                
                ax[2].set_title(f"output {count}")
                ax[2].imshow(out.argmax(0)[:, :, 80])

                fig.tight_layout()

                # Adding logs to TensorBoard
                tb_logger.experiment.add_figure(f"Image {count}", fig, self.current_epoch)
            
            else:
                continue

        # Adding logs to TensorBoard
        tb_logger.experiment.add_scalar("Validation Loss",mean_val_loss,self.current_epoch)
        tb_logger.experiment.add_scalar("Validation Dice",mean_val_dice,self.current_epoch)


        if mean_val_dice > self.best_val_dice:
            self.best_val_dice = mean_val_dice
            self.best_val_epoch = self.current_epoch

        print(
            f"\nEpoch [{self.current_epoch}/{self.max_epochs}], "
            f" Mean Validation Mean Dice: {mean_val_dice:.4f}"
            f" Best Validation Mean Dice: {self.best_val_dice:.4f} "
            f"at epoch: {self.best_val_epoch}"
            f' Validation Loss: {mean_val_loss:.4f}, '
        )
        self.validation_step_outputs.clear()
        return {"log":tensorboard_logs}
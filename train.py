import os
import pytorch_lightning 
from model import Lightning_Model
import torch 
import matplotlib.pyplot as plt
from monai.inferers import sliding_window_inference
from data import retrieve_data_from_link




if __name__=="__main__":
    
    retrieve_data_from_link()
    # initialise the LightningModule
    model = Lightning_Model()

    # set up loggers and checkpoints
    log_dir = os.path.join("/home/shirshak/3D-SPLEEN-MONAI-Pt-Lightning", "logs")
    tb_logger = pytorch_lightning.loggers.TensorBoardLogger(save_dir=log_dir)

    # initialise Lightning's trainer.
    trainer = pytorch_lightning.Trainer(
        devices=[0],
        max_epochs=600,
        logger=tb_logger,
        enable_checkpointing=True,
        num_sanity_val_steps=1,
        log_every_n_steps=16,
    )

    # train
    trainer.fit(model)
    print(f"train completed, best_metric: {model.best_val_dice:.4f} " f"at epoch {model.best_val_epoch}")

    # View training in tensorboard

    # %load_ext tensorboard
    # %tensorboard --logdir=$log_dir

    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    with torch.no_grad():
        for i, val_data in enumerate(model.val_dataloader()):
            roi_size = (160, 160, 160)
            sw_batch_size = 4
            val_outputs = sliding_window_inference(val_data["image"].to(device), roi_size, sw_batch_size, model)
            # plot the slice [:, :, 80]
            plt.figure("check", (18, 6))
            plt.subplot(1, 3, 1)
            plt.title(f"image {i}")
            plt.imshow(val_data["image"][0, 0, :, :, 80], cmap="gray")
            plt.subplot(1, 3, 2)
            plt.title(f"label {i}")
            plt.imshow(val_data["label"][0, 0, :, :, 80])
            plt.subplot(1, 3, 3)
            plt.title(f"output {i}")
            plt.imshow(torch.argmax(val_outputs, dim=1).detach().cpu()[0, :, :, 80])
            plt.show()



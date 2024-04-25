import os
import pytorch_lightning 
from model import Lightning_Model
import torch 
import matplotlib.pyplot as plt
from monai.inferers import sliding_window_inference
from data import retrieve_data_from_link


# For tensorboard tunneling 
# Please use 
# ssh -N -L 6007:127.0.0.1:6007 -i /home/shirshak/.ssh/id_ed25519 shirshak@124.41.198.88 

if __name__=="__main__":

    # Because of RuntimeError: received 0 items of ancdata
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))
    
    
    retrieve_data_from_link()

    max_epochs=5,
    
    # set up loggers and checkpoints
    log_dir = os.path.join("/home/shirshak/3D-SPLEEN-MONAI-Pt-Lightning", "logs")
    tb_logger = pytorch_lightning.loggers.TensorBoardLogger(save_dir=log_dir)


    # initialise Lightning's trainer.
    trainer = pytorch_lightning.Trainer(
        devices=[0],
        max_epochs=max_epochs,
        logger=tb_logger,
        enable_checkpointing=True,
        num_sanity_val_steps=1,
        log_every_n_steps=5,
    )

    # initialise the LightningModule
    model = Lightning_Model(tb_logger = tb_logger, max_epochs=max_epochs)


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
            fig, ax = plt.subplots(1,3)
            ax[0].set_title(f"image {i}")
            ax[0].imshow(val_data["image"][0, 0, :, :, 80], cmap="gray")
            
            ax[1].set_title(f"label {i}")
            ax[1].imshow(val_data["label"][0, 0, :, :, 80])
            
            ax[2].set_title(f"output {i}")
            ax[2].imshow(torch.argmax(val_outputs, dim=1).detach().cpu()[0, :, :, 80])

            fig.tight_layout()
            fig.savefig(f'images/img_label_output.png')





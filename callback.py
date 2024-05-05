from lightning.pytorch.callbacks import ModelCheckpoint

checkpoint = ModelCheckpoint(
    monitor='val_loss',
    dirpath='checkpoint',
    filename='brain-{epoch:02d}-{val_loss:.2f}', 
    save_top_k=1, 
    save_on_train_epoch_end=True,
    verbose=True, 
    save_weights_only=True
)


def get_callbacks() -> list: 
    return [
        checkpoint
    ]
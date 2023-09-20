import pytorch_lightning as pl


class PrintAccuracyAndLoss(pl.Callback):
    def on_epoch_end(self, trainer, pl_module):
        train_acc = pl_module.train_acc.compute()
        val_acc = pl_module.val_acc.compute()
        train_loss = trainer.callback_metrics["train_loss"]
        val_loss = trainer.callback_metrics["val_loss"]
        print(
            f"Epoch {trainer.current_epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, train_acc={train_acc:.4f}, val_acc={val_acc:.4f}"
        )

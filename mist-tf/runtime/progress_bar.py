from tensorflow.keras.utils import Progbar


class ProgressBar:

    def __init__(self, train_steps, val_steps, train_loss, val_loss):
        self.train_steps = train_steps
        self.val_steps = val_steps
        self.train_loss = train_loss
        self.val_loss = val_loss

        self.train_bar = Progbar(self.train_steps, stateful_metrics=['loss'])
        self.val_bar = Progbar(self.val_steps, stateful_metrics=['val_loss'])

    def update_train_bar(self):
        self.train_bar.add(1, values=[('loss', self.train_loss.result())])

    def update_val_bar(self):
        self.val_bar.add(1, values=[('val_loss', self.val_loss.result())])

    def reset(self):
        self.train_bar = Progbar(self.train_steps, stateful_metrics=['loss'])
        self.val_bar = Progbar(self.val_steps, stateful_metrics=['val_loss'])

        self.train_loss.reset_states()
        self.val_loss.reset_states()

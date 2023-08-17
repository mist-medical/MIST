import gc
import os.path


class Checkpoints:

    def __init__(self, best_model_path, last_model_path, best_value):

        self.best_model_path = os.path.abspath(best_model_path)
        self.last_model_path = os.path.abspath(last_model_path)
        self.best_value = best_value

    def save_last_model(self, last_model):
        print('Saving last model to {}'.format(self.last_model_path))
        last_model.save(self.last_model_path)
        gc.collect()

    def update(self, current_model, new_value):
        if new_value < self.best_value:
            print('*** Val loss IMPROVED from {:.4f} to {:.4f} ***'.format(self.best_value, new_value))
            print('Saving best model to {}'.format(self.best_model_path))
            self.best_value = new_value
            current_model.save(self.best_model_path)
        else:
            print('Val loss of DID NOT improve from {:.4f}'.format(self.best_value))

        gc.collect()



import json
import tensorflow.keras.backend as K

class Loss(object):
    
    def __init__(self, json_file):
        with open(json_file, 'r') as file:
            self.params = json.load(file)
            
        self.num_classes = len(self.params['labels'])
        self.smooth = 0.0000001
        
    def dice(self, y_true, y_pred):
        # (batch size, depth, height, width, channels)
        # skip the batch and class axis for calculating Dice score
        axes = tuple(range(1, len(y_true.shape) - 1))
        y_true = y_true[..., 0:self.num_classes]
        num = K.sum(K.square(y_true - y_pred), axis = axes)
        den = K.sum(K.square(y_true), axis = axes) + K.sum(K.square(y_pred), axis = axes) + self.smooth

        return K.mean(num/den, axis = -1)
        
    def gdl(self, y_true, y_pred):
        # (batch size, depth, height, width, channels)
        # skip the batch and class axis for calculating Dice score
        axes = tuple(range(1, len(y_true.shape) - 1))
    
        y_true = y_true[..., 0:self.num_classes]

        Wk = K.sum(y_true, axis = axes)
        Wk = 1. / (K.square(Wk) + 1.)

        num = K.sum(K.square(y_true - y_pred), axis = axes)
        den = K.sum(K.square(y_true), axis = axes) + K.sum(K.square(y_pred), axis = axes) + self.smooth

        return K.sum(Wk * num, axis = -1) / K.sum(Wk * den, axis = -1)
    
    def bl(self, y_true, y_pred):    
        dtm = y_true[..., self.num_classes:(2 * self.num_classes)]
        return K.mean(dtm * y_pred)
    
    def hdos(self, y_true, y_pred):
        # Separate y_true into distance transform and labels
        dtm = y_true[..., self.num_classes:(2 * self.num_classes)]
        y_true_labels = y_true[..., 0:self.num_classes]
        return K.mean(K.square(y_pred - y_true_labels) * K.square(dtm))
    
    def wnbl(self, y_true, y_pred):
        # (batch size, depth, height, width, channels)
        # skip the batch and class axis for calculating Dice score
        axes = tuple(range(1, len(y_true.shape) - 1))
    
        # Flip each one-hot encoded class
        y_worst = K.square(1 - y_true[..., 0:self.num_classes])

        # Separate y_true into distance transform and labels
        dtm = y_true[..., self.num_classes:(2 * self.num_classes)]
        y_true_labels = y_true[..., 0:self.num_classes]

        # Add weight for each class
        Wk = K.sum(y_true_labels, axis = axes)
        Wk = 1. / (K.square(Wk) + 1.)

        num = K.sum(K.square(dtm * (y_worst - y_pred)), axis = axes)
        den = K.sum(K.square(dtm * (y_worst - y_true_labels)), axis = axes) + self.smooth

        return 1 - (K.sum(Wk * num, axis = -1) / K.sum(Wk * den, axis = -1))
        
    def loss_wrapper(self, alpha):
        if self.params['loss'] == 'dice':
            def loss(y_true, y_pred):
                return self.dice(y_true, y_pred)
            
        elif self.params['loss'] == 'gdl':
            def loss(y_true, y_pred):
                return self.gdl(y_true, y_pred)
            
        elif self.params['loss'] == 'hdos':
            def loss(y_true, y_pred):
                return self.gdl(y_true, y_pred) + (1.0 - alpha) * self.hdos(y_true, y_pred)
            
        elif self.params['loss'] == 'bl':
            def loss(y_true, y_pred):
                return self.gdl(y_true, y_pred) + (1.0 - alpha) * self.bl(y_true, y_pred)
            
        elif self.params['loss'] == 'wnbl':
            def loss(y_true, y_pred):
                return self.gdl(y_true, y_pred) + (1.0 - alpha) * self.wnbl(y_true, y_pred)
            
        else:
            def loss(y_true, y_pred):
                return self.dice(y_true, y_pred)
        
        return loss
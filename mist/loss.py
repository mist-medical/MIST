import json
import tensorflow.keras.backend as K

class Loss(object):
    
    def __init__(self, json_file):
        with open(json_file, 'r') as file:
            self.params = json.load(file)
            
        self.num_classes = len(self.params['labels'])
        self.smooth = 0.0000001
        
    def dice_loss_weighted(self, y_true, y_pred):
        # (batch size, depth, height, width, channels)
        # skip the batch and class axis for calculating Dice score
        axes = tuple(range(1, len(y_true.shape) - 1))
    
        y_true = y_true[..., 0:self.num_classes]

        Wk = K.sum(y_true, axis = axes)
        Wk = 1. / (K.square(Wk) + 1.)

        num = K.sum(K.square(y_true - y_pred), axis = axes)
        den = K.sum(K.square(y_true), axis = axes) + K.sum(K.square(y_pred), axis = axes) + self.smooth

        return K.sum(Wk * num, axis = -1) / K.sum(Wk * den, axis = -1)
    
    def dice_loss(self, y_true, y_pred):
        # (batch size, depth, height, width, channels)
        # skip the batch and class axis for calculating Dice score
        axes = tuple(range(1, len(y_true.shape) - 1))
        y_true = y_true[..., 0:self.num_classes]
        num = K.sum(K.square(y_true - y_pred), axis = axes)
        den = K.sum(K.square(y_true), axis = axes) + K.sum(K.square(y_pred), axis = axes) + self.smooth

        return K.mean(num/den, axis = -1)
    
    def boundary_loss(self, y_true, y_pred):    
        dtm = y_true[..., self.num_classes:(2 * self.num_classes)]
        return K.mean(dtm * y_pred)
    
    def hd_loss_os(self, y_true, y_pred):
        # Separate y_true into distance transform and labels
        dtm = y_true[..., self.num_classes:(2 * self.num_classes)]
        y_true_labels = y_true[..., 0:self.num_classes]
        return K.mean(K.square(y_pred - y_true_labels) * K.square(dtm))
    
    def weighted_norm_boundary_loss(self, y_true, y_pred):
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
                return self.dice_loss(y_true, y_pred)
            
        elif self.params['loss'] == 'gdl':
            def loss(y_true, y_pred):
                return self.weighted_dice_loss(y_true, y_pred)
            
        elif self.params['loss'] == 'hdos':
            def loss(y_true, y_pred):
                return self.dice_loss_weighted(y_true, y_pred) + (1.0 - alpha) * self.hd_loss_os(y_true, y_pred)
            
        elif self.params['loss'] == 'bl':
            def loss(y_true, y_pred):
                return self.dice_loss_weighted(y_true, y_pred) + (1.0 - alpha) * self.boundary_loss(y_true, y_pred)
            
        elif self.params['loss'] == 'wnbl'
            def loss(y_true, y_pred):
                return self.dice_loss_weighted(y_true, y_pred) + (1.0 - alpha) * self.weighted_norm_boundary_loss(y_true, y_pred)
            
        else:
            def loss(y_true, y_pred):
                return self.dice_loss(y_true, y_pred)
        
        return loss
    
    
        
        

# def dice_loss_weighted(y_true, y_pred):
#     smooth = 0.0000001

#     # (batch size, depth, height, width, channels)
#     # skip the batch and class axis for calculating Dice score
#     axes = tuple(range(1, len(y_true.shape) - 1))
    
#     y_true = y_true[..., 0:4]

#     Wk = K.sum(y_true, axis = axes)
#     Wk = 1.0 / (K.square(Wk) + 1.0)

#     num = K.sum(K.square(y_true - y_pred), axis = axes)
#     den = K.sum(K.square(y_true), axis = axes) + K.sum(K.square(y_pred), axis = axes) + smooth

#     return K.sum(Wk * num, axis = -1) / K.sum(Wk * den, axis = -1)

# def dice_loss(y_true, y_pred):
#     smooth = 0.0000001

#     # (batch size, depth, height, width, channels)
#     # skip the batch and class axis for calculating Dice score
#     axes = tuple(range(1, len(y_true.shape) - 1))
#     y_true = y_true[..., 0:4]
#     num = K.sum(K.square(y_true - y_pred), axis = axes)
#     den = K.sum(K.square(y_true), axis = axes) + K.sum(K.square(y_pred), axis = axes) + smooth

#     return K.mean(num/den, axis = -1)

# def norm_surface_loss(y_true, y_pred):
#     smooth = 0.0000001
#     num_classes = 4
#     axes = tuple(range(1, len(y_true.shape) - 1))
    
#     # Flip each one-hot encoded class
#     y_worst = K.square(1.0 - y_true[..., 0:num_classes])
    
#     # Separate y_true into distance transform and labels
#     dtm = y_true[..., num_classes:(2 * num_classes)]
#     y_true_labels = y_true[..., 0:num_classes]
    
#     num = K.sum(K.square(dtm * (y_worst - y_pred)), axis = axes)
#     den = K.sum(K.square(dtm * (y_worst - y_true_labels)), axis = axes) + smooth
    
#     return 1.0 - K.mean(num/den, axis = -1)

# def surface_loss(y_true, y_pred):    
#     num_classes = 4
#     dtm = y_true[..., num_classes:(2 * num_classes)]
#     return K.mean(dtm * y_pred)

# def norm_surface_loss_weighted(y_true, y_pred):
#     smooth = 0.0000001
#     num_classes = 4
#     axes = tuple(range(1, len(y_true.shape) - 1))
    
#     # Flip each one-hot encoded class
#     y_worst = K.square(1 - y_true[..., 0:num_classes])
    
#     # Separate y_true into distance transform and labels
#     dtm = y_true[..., num_classes:(2 * num_classes)]
#     y_true_labels = y_true[..., 0:num_classes]
    
#     # Add weight for each class
#     Wk = K.sum(y_true_labels, axis = axes)
#     Wk = 1. / (K.square(Wk) + 1)
    
#     num = K.sum(K.square(dtm * (y_worst - y_pred)), axis = axes)
#     den = K.sum(K.square(dtm * (y_worst - y_true_labels)), axis = axes) + smooth

#     return 1 - (K.sum(Wk * num, axis = -1) / K.sum(Wk * den, axis = -1))

# def dice_norm_surf_loss_wrapper(alpha):
#     def dice_norm_surf_loss(y_true, y_pred):
#         return dice_loss_weighted(y_true, y_pred) + (1.0 - alpha) * norm_surface_loss_weighted(y_true, y_pred)
#     return dice_norm_surf_loss
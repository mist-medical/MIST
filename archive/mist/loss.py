import json
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

class DiceLoss(tf.keras.losses.Loss):
    def __init__(self, **kwargs):
        super().__init__()
        self.axes = (1, 2, 3)
        self.smooth = 1.e-6
        
    def call(self, y_true, y_pred):
        num = tf.reduce_sum(tf.square(y_true - y_pred), axis = self.axes)
        den = tf.reduce_sum(tf.square(y_true), axis = self.axes) + \
              tf.reduce_sum(K.square(y_pred), axis = self.axes) + self.smooth
        return tf.reduce_mean(num / den, axis = -1)
    
class DiceCELoss(tf.keras.losses.Loss):
    def __init__(self, **kwargs):
        super().__init__()
        self.axes = (1, 2, 3)
        self.smooth = 1.e-6
        self.dice_loss = DiceLoss()
        
    def call(self, y_true, y_pred):
        dice_loss = self.dice_loss(y_true, y_pred)
        ce_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_true, logits = y_pred))
        return dice_loss + ce_loss
    
class WeightedDiceLoss(tf.keras.losses.Loss):
    def __init__(self, class_weights, **kwargs):
        super().__init__()
        self.class_weights = class_weights
        self.axes = (1, 2, 3)
        self.smooth = 1.e-6
        
    def call(self, y_true, y_pred):
        num = tf.reduce_sum(K.square(y_true - y_pred), axis = self.axes)
        num *= self.class_weights

        den = tf.reduce_sum(tf.square(y_true), axis = self.axes) + \
              tf.reduce_sum(tf.square(y_pred), axis = self.axes)
        den *= self.class_weights
        den += self.smooth

        return tf.reduce_sum(num, axis = -1) / tf.reduce_sum(den, axis = -1)
    
    def get_config(self):
        return {'class_weights': self.class_weights}
    
class BoundaryLoss(tf.keras.losses.Loss):
    def __init__(self, class_weights, alpha, **kwargs):
        super().__init__()
        self.class_weights = class_weights
        self.alpha = alpha
        self.axes = (1, 2, 3)
        self.smooth = 1.e-6
        self.gdl = WeightedDiceLoss(self.class_weights)
        
    def call(self, y_true, y_pred):
        n_classes = K.shape(y_true)[-1] // 2
        dtm = y_true[..., n_classes:(2 * n_classes)]
        bl = tf.reduce_mean(dtm * y_pred)
        return self.alpha * self.gdl(y_true[..., 0:n_classes], y_pred) + (1.0 - self.alpha) * bl
    
    def get_config(self):
        return {'class_weights': self.class_weights, 
                'alpha': self.alpha}
    
class HDOSLoss(tf.keras.losses.Loss):
    def __init__(self, class_weights, alpha, **kwargs):
        super().__init__()
        self.class_weights = class_weights
        self.alpha = alpha
        self.axes = (1, 2, 3)
        self.smooth = 1.e-6
        self.gdl = WeightedDiceLoss(self.class_weights)
        
    def call(self, y_true, y_pred):
        n_classes = K.shape(y_true)[-1] // 2
        dtm = y_true[..., n_classes:(2 * n_classes)]
        hdos = tf.reduce_mean(tf.square(y_pred - y_true[..., 0:n_classes]) * tf.square(dtm))
        return self.alpha * self.gdl(y_true[..., 0:n_classes], y_pred) + (1.0 - self.alpha) * hdos
    
    def get_config(self):
        return {'class_weights': self.class_weights, 
                'alpha': self.alpha}
    
class WNBLoss(tf.keras.losses.Loss):
    def __init__(self, class_weights, alpha, **kwargs):
        super().__init__()
        self.class_weights = class_weights
        self.alpha = alpha
        self.axes = (1, 2, 3)
        self.smooth = 1.e-6
        self.gdl = WeightedDiceLoss(self.class_weights)
        
    def call(self, y_true, y_pred):
        # Get number of classes
        n_classes = K.shape(y_true)[-1] // 2
        
        # Flip each one-hot encoded class
        y_worst = K.square(1.0 - y_true[..., 0:n_classes])

        # Separate y_true into distance transform and labels
        dtm = y_true[..., n_classes:(2 * n_classes)]
        y_true_labels = y_true[..., 0:n_classes]

        num = tf.reduce_sum(tf.square(dtm * (y_worst - y_pred)), axis = self.axes)
        num *= self.class_weights

        den = tf.reduce_sum(tf.square(dtm * (y_worst - y_true_labels)), axis = self.axes)
        den *= self.class_weights
        den += self.smooth

        return 1.0 - (tf.reduce_sum(num, axis = -1) / tf.reduce_sum(den, axis = -1))
    
    def get_config(self):
        return {'class_weights': self.class_weights, 
                'alpha': self.alpha}
    
def get_loss(params, **kwargs):
    if 'loss' in params.keys():
        if params['loss'] == 'dice':
            loss_fn = DiceLoss()
            custom_object = {'DiceLoss': loss_fn}
            
        if params['loss'] == 'dice_ce':
            loss_fn = DiceCELoss()
            custom_object = {'DiceCELoss': loss_fn}

        elif params['loss'] == 'gdl':
            loss_fn = WeightedDiceLoss(class_weights = kwargs['class_weights'])
            custom_object = {'WeightedDiceLoss': loss_fn}

        elif params['loss'] == 'bl':
            loss_fn = BoundaryLoss(class_weights = kwargs['class_weights'], alpha = kwargs['alpha'])
            custom_object = {'BoundaryLoss': loss_fn}

        elif params['loss'] == 'hdos':
            loss_fn = HDOSLoss(class_weights = kwargs['class_weights'], alpha = kwargs['alpha'])
            custom_object = {'HDOSLoss': loss_fn}

        elif params['loss'] == 'wnbl':
            loss_fn = WNBLoss(class_weights = kwargs['class_weights'], alpha = kwargs['alpha'])
            custom_object = {'WNBLoss': loss_fn}

        else:
            loss_fn = DiceLoss()
            custom_object = {'DiceLoss': loss_fn}
    else:
        loss_fn = DiceLoss()
        custom_object = {'DiceLoss': loss_fn}
        
    return loss_fn, custom_object
        
    
# class MISTLosses(tf.keras.losses.Loss):
    
#     def __init__(self, params, **kwargs):
#         super().__init__()

#         self.params = params

#         with open(self.params['inferred_params'], 'r') as file:
#             self.inferred_params = json.load(file)

#         self.class_weights = tf.constant(np.array(self.inferred_params['class_weights'], dtype=np.float32))
            
#         self.num_classes = len(self.params['labels'])
        
#         self.num_classes = num_classes
#         self.class_weights = tf.constant(np.array(kwargs['class_weights'], dtype = np.float32))
#         self.alpha = kwargs['alpha']
#         self.smooth = 1.e-6
        
#     def dice(self, y_true, y_pred):
#         # (batch size, depth, height, width, channels)
#         # skip the batch and class axis for calculating Dice score
#         axes = tuple(range(1, len(y_true.shape) - 1))
#         y_true = y_true[..., 0:self.num_classes]
#         num = K.sum(K.square(y_true - y_pred), axis = axes)
#         den = K.sum(K.square(y_true), axis = axes) + K.sum(K.square(y_pred), axis = axes) + self.smooth

#         return K.mean(num/den, axis = -1)
        
#     def gdl(self, y_true, y_pred):
#         # (batch size, depth, height, width, channels)
#         # skip the batch and class axis for calculating Dice score
#         axes = tuple(range(1, len(y_true.shape) - 1))
    
#         y_true = y_true[..., 0:self.num_classes]

#         num = K.sum(K.square(y_true - y_pred), axis = axes)
#         num *= self.class_weights

#         den = K.sum(K.square(y_true), axis = axes) + K.sum(K.square(y_pred), axis = axes)
#         den *= self.class_weights
#         den += self.smooth

#         return K.sum(num, axis = -1) / K.sum(den, axis = -1)
    
#     def bl(self, y_true, y_pred):    
#         dtm = y_true[..., self.num_classes:(2 * self.num_classes)]
#         return K.mean(dtm * y_pred)
    
#     def hdos(self, y_true, y_pred):
#         # Separate y_true into distance transform and labels
#         dtm = y_true[..., self.num_classes:(2 * self.num_classes)]
#         y_true_labels = y_true[..., 0:self.num_classes]
#         return K.mean(K.square(y_pred - y_true_labels) * K.square(dtm))
    
#     def wnbl(self, y_true, y_pred):
#         # (batch size, depth, height, width, channels)
#         # skip the batch and class axis for calculating Dice score
#         axes = tuple(range(1, len(y_true.shape) - 1))
    
#         # Flip each one-hot encoded class
#         y_worst = K.square(1 - y_true[..., 0:self.num_classes])

#         # Separate y_true into distance transform and labels
#         dtm = y_true[..., self.num_classes:(2 * self.num_classes)]
#         y_true_labels = y_true[..., 0:self.num_classes]

#         num = K.sum(K.square(dtm * (y_worst - y_pred)), axis = axes)
#         num *= self.class_weights

#         den = K.sum(K.square(dtm * (y_worst - y_true_labels)), axis = axes)
#         den *= self.class_weights
#         den += self.smooth

#         return 1. - (K.sum(num, axis = -1) / K.sum(den, axis = -1))

#     def call(self, y_true, y_pred):        
#         if self.params['loss'] == 'dice':
#             return self.dice(y_true, y_pred)
            
#         elif self.params['loss'] == 'gdl':
#             return self.gdl(y_true, y_pred)
            
#         elif self.params['loss'] == 'hdos':
#             return self.alpha * self.gdl(y_true, y_pred) + (1.0 - self.alpha) * self.hdos(y_true, y_pred)
            
#         elif self.params['loss'] == 'bl':
#             return self.alpha * self.gdl(y_true, y_pred) + (1.0 - self.alpha) * self.bl(y_true, y_pred)
            
#         elif self.params['loss'] == 'wnbl':
#             return self.alpha * self.gdl(y_true, y_pred) + (1.0 - self.alpha) * self.wnbl(y_true, y_pred)
            
#         else:
#             return self.dice(y_true, y_pred)
                
#     def get_config(self):
#         return {'num_classes': self.self.num_classes, 
#                 'class_weights': kwargs['class_weights'], 
#                 'alpha': kwargs['alpha']}
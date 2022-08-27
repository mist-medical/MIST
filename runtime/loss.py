import tensorflow as tf


def get_one_hot(y_true, y_pred):
    y_true = tf.one_hot(y_true, y_pred.shape[-1])
    y_true = tf.squeeze(y_true, 4)
    y_true = tf.cast(y_true, tf.float32)
    return y_true


class DiceLoss(tf.keras.losses.Loss):
    def __init__(self, **kwargs):
        super(DiceLoss, self).__init__()
        self.axes = (1, 2, 3)
        self.smooth = 1.e-6

    def call(self, y_true, y_pred):
        # Make inputs compatible with loss calculation
        y_true = get_one_hot(y_true, y_pred)
        y_pred = tf.nn.softmax(y_pred)

        num = tf.reduce_sum(tf.square(y_true - y_pred), axis=self.axes)
        den = tf.reduce_sum(tf.square(y_true), axis=self.axes) + tf.reduce_sum(tf.square(y_pred),
                                                                               axis=self.axes) + self.smooth
        return tf.reduce_mean(num / den, axis=-1)


class DiceCELoss(tf.keras.losses.Loss):
    def __init__(self, **kwargs):
        super(DiceCELoss, self).__init__()
        self.axes = (1, 2, 3)
        self.smooth = 1.e-6
        self.dice_loss = DiceLoss()

    def call(self, y_true, y_pred):
        dice_loss = self.dice_loss(y_true, y_pred)

        y_true = get_one_hot(y_true, y_pred)
        ce_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred))
        return dice_loss + ce_loss


class WeightedDiceLoss(tf.keras.losses.Loss):
    def __init__(self, class_weights, **kwargs):
        super(WeightedDiceLoss, self).__init__()
        self.class_weights = class_weights
        self.axes = (1, 2, 3)
        self.smooth = 1.e-6

    def call(self, y_true, y_pred):
        # Make inputs compatible with loss calculation
        y_true = get_one_hot(y_true, y_pred)
        y_pred = tf.nn.softmax(y_pred)

        num = tf.reduce_sum(tf.square(y_true - y_pred), axis=self.axes)
        num *= self.class_weights

        den = tf.reduce_sum(tf.square(y_true), axis=self.axes) + tf.reduce_sum(tf.square(y_pred), axis=self.axes)
        den *= self.class_weights
        den += self.smooth

        return tf.reduce_sum(num, axis=-1) / tf.reduce_sum(den, axis=-1)

    def get_config(self):
        return {'class_weights': self.class_weights}


class WeightedDiceCELoss(tf.keras.losses.Loss):
    def __init__(self, class_weights, **kwargs):
        super(WeightedDiceCELoss, self).__init__()
        self.class_weights = class_weights
        self.axes = (1, 2, 3)
        self.smooth = 1.e-6
        self.gdl_loss = WeightedDiceLoss(class_weights)

    def call(self, y_true, y_pred):
        gdl_loss = self.gdl_loss(y_true, y_pred)

        y_true = get_one_hot(y_true, y_pred)
        ce_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred))
        return gdl_loss + ce_loss

    def get_config(self):
        return {'class_weights': self.class_weights}


# class BoundaryLoss(tf.keras.losses.Loss):
#     def __init__(self, class_weights, **kwargs):
#         super(BoundaryLoss, self).__init__()
#         self.class_weights = class_weights
#         self.axes = (1, 2, 3)
#         self.smooth = 1.e-6
#         self.gdl = WeightedDiceLoss(self.class_weights)
#
#     def call(self, y_true, y_pred):
#         y_true = tf.cast(y_true, tf.float32)
#         n_classes = K.shape(y_true)[-1] // 2
#         dtm = y_true[..., n_classes:(2 * n_classes)]
#         bl = tf.reduce_mean(dtm * y_pred)
#         return self.gdl(y_true[..., 0:n_classes], y_pred) + bl
#
#     def get_config(self):
#         return {'class_weights': self.class_weights}
#
#
# class WNBLoss(tf.keras.losses.Loss):
#     def __init__(self, class_weights, **kwargs):
#         super(WNBLoss, self).__init__()
#         self.class_weights = class_weights
#         self.axes = (1, 2, 3)
#         self.smooth = 1.e-6
#         self.gdl = WeightedDiceLoss(self.class_weights)
#
#     def call(self, y_true, y_pred):
#         y_true = tf.cast(y_true, tf.float32)
#
#         # Get number of classes
#         n_classes = K.shape(y_true)[-1] // 2
#
#         # Flip each one-hot encoded class
#         y_worst = tf.square(1.0 - y_true[..., 0:n_classes])
#
#         # Separate y_true into distance transform and labels
#         dtm = y_true[..., n_classes:(2 * n_classes)]
#         y_true_labels = y_true[..., 0:n_classes]
#
#         num = tf.reduce_sum(tf.square(dtm * (y_worst - y_pred)), axis=self.axes)
#         num *= self.class_weights
#
#         den = tf.reduce_sum(tf.square(dtm * (y_worst - y_true_labels)), axis=self.axes)
#         den *= self.class_weights
#         den += self.smooth
#
#         wnbl = 1.0 - (tf.reduce_sum(num, axis=-1) / tf.reduce_sum(den, axis=-1))
#
#         return self.gdl(y_true[..., 0:n_classes], y_pred) + wnbl
#
#     def get_config(self):
#         return {'class_weights': self.class_weights}


def get_loss(args, **kwargs):
    if args.loss == 'dice':
        loss_fn = DiceLoss()

    elif args.loss == 'dice_ce':
        loss_fn = DiceCELoss()

    elif args.loss == 'gdl':
        loss_fn = WeightedDiceLoss(class_weights=kwargs['class_weights'])

    elif args.loss == 'gdl_ce':
        loss_fn = WeightedDiceCELoss(class_weights=kwargs['class_weights'])

    else:
        loss_fn = DiceCELoss()

    return loss_fn

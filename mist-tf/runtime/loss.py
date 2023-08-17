import tensorflow as tf


def get_one_hot(y_true, y_pred):
    y_true = tf.one_hot(y_true, y_pred.shape[-1])
    y_true = tf.squeeze(y_true, 4)
    y_true = tf.cast(y_true, tf.float32)
    return y_true


class DiceLoss(tf.keras.losses.Loss):
    def __init__(self, **kwargs):
        super(DiceLoss, self).__init__(reduction=kwargs["reduction"])
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
        super(DiceCELoss, self).__init__(reduction=kwargs["reduction"])
        self.axes = (1, 2, 3)
        self.smooth = 1.e-6
        self.dice_loss = DiceLoss(**kwargs)

    def call(self, y_true, y_pred):
        dice_loss = self.dice_loss(y_true, y_pred)

        y_true = get_one_hot(y_true, y_pred)
        ce_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred))
        return dice_loss + ce_loss


class WeightedDiceLoss(tf.keras.losses.Loss):
    def __init__(self, class_weights, **kwargs):
        super(WeightedDiceLoss, self).__init__(reduction=kwargs["reduction"])
        self.class_weights = class_weights
        self.axes = (1, 2, 3)
        self.smooth = 1.e-6

    def call(self, y_true, y_pred):
        # Make inputs compatible with loss calculation
        y_true = get_one_hot(y_true, y_pred)
        y_pred = tf.nn.softmax(y_pred)

        if self.class_weights is None:
            class_weights = tf.reduce_sum(y_true, axis=self.axes)
            class_weights = 1. / (tf.square(class_weights) + 1.)
        else:
            class_weights = self.class_weights

        num = tf.reduce_sum(tf.square(y_true - y_pred), axis=self.axes)
        num *= class_weights

        den = tf.reduce_sum(tf.square(y_true), axis=self.axes) + tf.reduce_sum(tf.square(y_pred), axis=self.axes)
        den *= class_weights
        den += self.smooth

        return tf.reduce_sum(num, axis=-1) / tf.reduce_sum(den, axis=-1)

    def get_config(self):
        return {'class_weights': self.class_weights}


class WeightedDiceCELoss(tf.keras.losses.Loss):
    def __init__(self, class_weights, **kwargs):
        super(WeightedDiceCELoss, self).__init__(reduction=kwargs["reduction"])
        self.class_weights = class_weights
        self.axes = (1, 2, 3)
        self.smooth = 1.e-6
        self.gdl_loss = WeightedDiceLoss(class_weights, **kwargs)

    def call(self, y_true, y_pred):
        gdl_loss = self.gdl_loss(y_true, y_pred)

        y_true = get_one_hot(y_true, y_pred)
        ce_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred))
        return gdl_loss + ce_loss

    def get_config(self):
        return {'class_weights': self.class_weights}


def get_loss(args, **kwargs):
    if args.loss == 'dice':
        loss_fn = DiceLoss(reduction=kwargs["reduction"])

    elif args.loss == 'dice_ce':
        loss_fn = DiceCELoss(reduction=kwargs["reduction"])

    elif args.loss == 'gdl':
        loss_fn = WeightedDiceLoss(class_weights=kwargs['class_weights'], reduction=kwargs["reduction"])

    elif args.loss == 'gdl_ce':
        loss_fn = WeightedDiceCELoss(class_weights=kwargs['class_weights'], reduction=kwargs["reduction"])

    else:
        loss_fn = DiceCELoss(reduction=kwargs["reduction"])

    return loss_fn

# L2 Dice loss
def dice_loss(y_true, y_pred):
    smooth = 0.0000001

    # (batch size, depth, height, width, channels)
    # skip the batch and class axis for calculating Dice score
    axes = tuple(range(1, len(y_true.shape) - 1))
    num = K.sum(K.square(y_true - y_pred), axis = axes)
    den = K.sum(K.square(y_true), axis = axes) + K.sum(K.square(y_pred), axis = axes) + smooth

    return K.mean(num/den, axis = -1)

def dice_loss_weighted(y_true, y_pred):
    smooth = 0.0000001

    # (batch size, depth, height, width, channels)
    # skip the batch and class axis for calculating Dice score
    axes = tuple(range(1, len(y_true.shape) - 1))

    Wk = K.sum(y_true, axis = axes)
    Wk = Wk * K.square(1. / (Wk + 1.))

    num = K.sum(K.square(y_true - y_pred), axis = axes)
    den = K.sum(K.square(y_true), axis = axes) + K.sum(K.square(y_pred), axis = axes) + smooth

    return K.sum(Wk * num, axis = -1) / K.sum(Wk * den, axis = -1)

# L2 Dice loss
def dice_loss_l2(y_true, y_pred):
    smooth = 0.0000001

    # (batch size, depth, height, width, channels)
    # skip the batch and class axis for calculating Dice score
    axes = tuple(range(1, len(y_true.shape) - 1))
    num = K.sum(K.square(y_true - y_pred), axis = axes)
    den = K.sum(K.square(y_true), axis = axes) + K.sum(K.square(y_pred), axis = axes) + smooth

    return K.mean(num/den, axis = -1)

# L2 Dice loss
def dice_loss_soft(y_true, y_pred):
    smooth = 0.0000001

    # (batch size, depth, height, width, channels)
    # skip the batch and class axis for calculating Dice score
    axes = tuple(range(1, len(y_true.shape) - 1))
    num = 2 * K.sum(y_true * y_pred, axis = axes) + smooth
    den = K.sum(K.square(y_true), axis = axes) + K.sum(K.square(y_pred), axis = axes) + smooth

    return 1 - K.mean(num/den, axis = -1)

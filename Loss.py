
EPSILON = 1e-10

def mse_loss(esti_list, label, mask_for_loss):
    masked_esti = esti_list * mask_for_loss
    masked_label = label * mask_for_loss
    loss = ((masked_esti - masked_label) ** 2).sum() / mask_for_loss.sum() + EPSILON
    return loss
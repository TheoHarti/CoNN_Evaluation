import torch


# def covariance(prediction, error_correlation):
#     prediction_mean = torch.mean(input=prediction, dim=0)
#     # We want to try to maximize the absolute covariance, but pytorch will minimize the loss function
#     # Therefore, we need to multiply by (-1) to guide the optimizer correctly
#     loss = -torch.sum(torch.abs(input=torch.sum(input=((prediction - prediction_mean) * error_correlation), dim=0)), dim=0)
#     return loss

def covariance(hidden_node_predictions, errors):
    value_term = hidden_node_predictions - torch.mean(input=hidden_node_predictions, dim=0)
    error_term = errors - torch.mean(input=errors, dim=0)
    output_covariances = torch.mm(value_term.T, error_term)
    complete_covariance = -torch.sum(torch.abs(output_covariances), dim=1)
    return complete_covariance

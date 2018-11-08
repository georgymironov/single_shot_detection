import torch


def match_bipartite(weights, inplace=False):
    """
    Args:
        weights: torch.tensor(:shape [Boxes, AnchorBoxes])
    Returns:
        box_idx: torch.tensor(:shape [Boxes])
        anchor_idx: torch.tensor(:shape [Boxes])
    """
    assert weights.max(dim=1)[0].gt(0).all().item()

    if not inplace:
        weights = weights.clone()

    num_boxes, num_priors = weights.size()

    box_idx = torch.arange(num_boxes, dtype=torch.long, device=weights.device)
    anchor_idx = torch.empty((num_boxes,), dtype=torch.long, device=weights.device)

    for _ in range(num_boxes):
        idx = weights.argmax()
        anchor_idx[idx // num_priors] = idx % num_priors
        weights[:, idx % num_priors] = 0
        weights[idx // num_priors] = 0

    return box_idx, anchor_idx

def match_per_prediction(weights, threshold):
    """
    Args:
        weights: torch.tensor(:shape [Boxes, AnchorBoxes])
    Returns:
        box_idx: torch.tensor(:shape [MathchedAnchors])
        anchor_idx: torch.tensor(:shape [MathchedAnchors])
    """
    overlap, box_idx = weights.max(dim=0)
    anchor_idx = torch.nonzero(overlap > threshold)
    return box_idx[anchor_idx], anchor_idx

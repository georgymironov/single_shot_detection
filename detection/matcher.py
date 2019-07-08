import torch


NOT_MATCHED = -2
IGNORE = -1

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

def match_per_prediction(weights, matched_threshold, unmatched_threshold=None, force_match_for_each_target=True):
    """
    Args:
        weights: torch.tensor(:shape [Boxes, AnchorBoxes])
    Returns:
        box_idx: torch.tensor(:shape [AnchorBoxes])
    """
    if unmatched_threshold is None:
        unmatched_threshold = matched_threshold
    else:
        assert matched_threshold >= unmatched_threshold

    overlap, box_idx = weights.max(dim=0)
    below_matched = overlap < matched_threshold
    below_unmatched = overlap < unmatched_threshold

    box_idx[below_unmatched] = NOT_MATCHED
    box_idx[below_matched & ~below_unmatched] = IGNORE

    if force_match_for_each_target:
        anchor_idx = weights.argmax(dim=1)
        box_idx[anchor_idx] = torch.arange(0, anchor_idx.size(0), dtype=box_idx.dtype, device=box_idx.device)

    return box_idx

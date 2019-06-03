

class _AnchorGenerator(object):
    def _generate_anchors(self, img_size, feature_map_size, device='cpu'):
        raise NotImplementedError

    def generate(self, img, feature_map):
        """
        Args:
            img: torch.tensor(:shape [Batch, Channels, Height, Width])
            feature_map: torch.tensor(:shape [Batch, Channels, Height, Width])
        Returns:
            priors: torch.tensor(:shape [Height, Width, AspectRatios, 4])
        """
        img_size = img.size(3), img.size(2)
        feature_map_size = feature_map.size(3), feature_map.size(2)

        anchors = self._generate_anchors(img_size, feature_map_size)

        return anchors

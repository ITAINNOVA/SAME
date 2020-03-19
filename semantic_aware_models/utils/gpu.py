import torch

class GPU:
    """ Class to work with GPUs """
    def __init__(self):
        pass

    def get_default_device(self):

        """
        Get the default device, if cuda is available get cuda.
        """
        if torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')

    def to_device(self, data, device):

        """
        Move data to default device.
        """

        if isinstance(data, (list,tuple)):
            return [self.to_device(x, device) for x in data]
        return data.to(device, non_blocking=True)


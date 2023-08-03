import os
from .imagenet import ImageNet


class ImageNetC(ImageNet):

    def populate_train(self):
        pass

    def get_test_path(self):
        return os.path.join(self.location)

from lolograd.layers.conv import Convolutional2DLayer
from lolograd.layers.dense import DenseLayer
from lolograd.layers.neuron import Neuron


class SimpleCNN:
    def __init__(
        self, input_size, input_depth, num_classes, conv_num_filters, conv_filter_size=3
    ):
        self.input_size = input_size
        self.conv_layer = Convolutional2DLayer(
            conv_num_filters, conv_filter_size, input_depth
        )

        # Assuming the output of the conv layer is flattened
        # The size of the flattened vector needs to be calculated based on the input size, filter size, and the number of filters
        flattened_size = conv_num_filters * (input_size - conv_filter_size + 1) ** 2
        self.output_layer = DenseLayer(flattened_size, num_classes)

    def __call__(self, input_volume):
        conv_output = self.conv_layer(input_volume)
        flattened = [x for row in conv_output for x in row]
        return self.output_layer(flattened)

    def parameters(self):
        return self.output_layer.parameters() + self.conv_layer.parameters()

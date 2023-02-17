import argparse
from collections import OrderedDict

import torch

from aitemplate.compiler import compile_model
from aitemplate.frontend import nn, Tensor
from aitemplate.testing import detect_target
from aitemplate.testing.benchmark_pt import benchmark_torch_function
from aitemplate.utils.graph_utils import sorted_graph_pseudo_code
from convperf.python.reader import load_csv, get_config, get_labels

p = argparse.ArgumentParser()

p.add_argument(
    "--precision",
    type=str,
    default="f32",
    help="f16 or f32",
)

args = p.parse_args()

# nn.Conv2d(16, 33, 3, 2)
# def __init__(
#        self,
#        in_channels,
#        out_channels,
#        kernel_size,
#        stride,
#        padding=0,
#        dilation=1,
#        groups=1,
#        dtype="float16",
#    )
# input = Tensor(shape=[20, 50, 100, 16])

class PTSimpleModel(torch.nn.Module):
    def __init__(self, c_in, c_out, kH, kW, strides, padding=0, dilation=1):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(c_in, c_out, (kH,kW), strides, padding, dilation)

    def forward(self, input):
        hidden_states = self.conv1(input)
        return hidden_states


class AITConv2d(nn.Module):
    def __init__(self, c_in, c_out, kH, kW, strides, padding=0, dilation=1, data_type="float16"):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, kH, strides, padding, dilation, dtype=data_type)

    def forward(self, input):
        hidden_states = self.conv1(input)
        return hidden_states

def map_pt_params(ait_model, pt_model):
    ait_model.name_parameter_tensor()
    pt_params = dict(pt_model.named_parameters())
    mapped_pt_params = OrderedDict()
    for name, _ in ait_model.named_parameters():
        ait_name = name.replace(".", "_")
        assert name in pt_params
        params = pt_params[name].permute((0,3,1,2)).contiguous()
        mapped_pt_params[ait_name] = params
        print(ait_name, _.tensor().shape, pt_params[name].shape)
    return mapped_pt_params

def run_conv(batch_size, h, w, c_in, c_out, kW, kH, strides, padding, dilations, data_type):
    pt_model = PTSimpleModel(c_in, c_out, kH, kW, strides, padding, dilations).cuda()
    if data_type == "float16":
        pt_model = pt_model.half()
    x = torch.randn([batch_size, c_in, h, w]).cuda()
    if data_type == "float16":
        x = x.half()

    pt_model.eval()
    y_pt = pt_model(x)

    ait_model = AITConv2d(c_in, c_out, kH, kW, strides, padding, dilations, data_type)
    weights = map_pt_params(ait_model, pt_model)
#    exit(0)

    X = Tensor(
        shape=[batch_size, h, w, c_in],
        name="X",
        dtype=data_type,
        is_input=True,
    )
    Y = ait_model(X)
    Y._attrs["is_output"] = True
    Y._attrs["name"] = "Y"
    # y_size = [dim.value() for dim in Y._attrs["shape"]]

    target = detect_target()

    with compile_model(Y, target, "./tmp", "conv", constants=weights) as module:
        y = torch.empty((y_pt.shape[0], y_pt.shape[-1],
                         y_pt.shape[-2], y_pt.shape[1])).cuda()
        if data_type == "float16":
            y = y.half()
        x = x.permute((0,2,3,1)).contiguous()
        inputs = {"X": x}
        outputs = {"Y": y}
#

        # module.run_with_tensors(inputs, outputs, graph_mode=True)

        count = 100
        ait_t, _, _ = module.benchmark_with_tensors(
            inputs, outputs, graph_mode=True, count=count
        )
        print(f"AITemplate time: {ait_t} ms/iter")
        return ait_t

data_types = {"f32" : "float32",
              "f16" : "float16",}

def benchmark_conv():
    data = load_csv('../benchmark_sizes/sd.csv')

    times = []
    labels = []

    data_type = args.precision
    dtype = data_types[data_type]

    for config in data['configs']:
        input, filter, output, strides, padding, dilation = get_config(config)
        # ait constraint
        assert(filter['H'] == filter['W'])
        time = run_conv(batch_size=input['N'], h=input['H'], w=input['W'], c_in=input['C'], c_out=filter['F'],
                        kW=filter['W'], kH=filter['H'],
                        strides=strides, padding=padding, dilations=dilation, data_type=dtype)

        label = get_labels(input, filter, output)
        label['strides'] = strides
        labels.append(label)
        times.append(time)

    header = 'op, input_shape, filter_shape, output_shape, stride, dtype, ms'
    print(header)
    for label, t in zip(labels, times):
        line = '{}, {}, {}, {}, {}, {}, {}'.format(label['op'], label['input_shape'], label['filter_shape'], \
            label['output_shape'], label['strides'], data_type, t)
        print(line)

benchmark_conv()
# run_conv(20, 50, 100, 16)

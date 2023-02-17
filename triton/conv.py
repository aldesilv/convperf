import argparse
from convperf.python.reader import load_csv, get_config, get_labels

import torch
from torch._inductor.utils import has_triton
from torch._inductor.triton_ops.conv import _conv
import triton

p = argparse.ArgumentParser()

p.add_argument(
    "--precision",
    type=str,
    default="f32",
    help="f16 or f32",
)

args = p.parse_args()

def run_conv(n, c_in, h, w, c_out, kH, kW, strides, padding, dilations, data_type):
    input = torch.randn([n, c_in, h, w]).cuda().to(data_type)
    weight = torch.randn([c_out, c_in, kH, kW]).cuda().to(data_type)
    bias = torch.randn([c_out]).cuda().to(data_type)
    print(input.dtype)
    conv1 = _conv()
    ms, min_ms, max_ms = triton.testing.do_bench(
                             lambda: conv1.forward(input, weight, bias,
                                                   stride=strides, padding=padding,
                                                   dilation=dilations),
                             rep=100)
    return ms
#    import triton._C.libtriton.triton as _triton
#    backend = _triton.runtime.backend.CUDA
#    device = torch.cuda.current_device()
#    cc = _triton.runtime.cc(backend, device)

data_types = {"f32" : torch.float32,
              "f16" : torch.float16,}

data = load_csv('../benchmark_sizes/sd.csv')
times = []
labels = []
data_type = args.precision
dtype = data_types[data_type]


for config in data['configs']:
    input, filter, output, strides, padding, dilation = get_config(config)
    time = run_conv(n=input['N'], c_in=input['C'], h=input['H'], w=input['W'], c_out=filter['F'],
                    kH=filter['H'], kW=filter['W'], strides=(strides, strides),
                    padding=(padding, padding), dilations=(dilation, dilation),
                    data_type=dtype)
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
# run_conv(2,4,66,66,320,3,3)

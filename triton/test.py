import argparse
import json
import time
import google_benchmark as benchmark
from convperf.python.reader import load_csv, get_config, get_labels

import torch

import torch._dynamo
import torch._inductor
from torch._inductor.utils import has_triton

p = argparse.ArgumentParser()

p.add_argument(
    "--precision",
    type=str,
    default="f32",
    help="f16 or f32",
)

args = p.parse_args()

# python test.py
# gpu

def test_conv_extern_kernel(N, C_in, C_out, H, W, kW, kH, strides, padding=0, dilations=1, data_type=torch.float32):
    class M(torch.nn.Module):
        def __init__(
            self,
            **kwargs,
        ):
            super().__init__()
            # in_c, out_c, kernel_size, stride, padding, dilation, ...
            self.conv = torch.nn.Conv2d(
                C_in,
                C_out,
                (kH, kW),
                strides,
                padding,
                dilations,
                **kwargs,
            )

        def forward(self, x):
            x1 = self.conv(x)
        #    print(x1.shape)
            return x1

    # if state is None:
    #    mod = M()
    #    opt_mod = torch._dynamo.optimize("inductor")(mod)
    #    return

#    while state:
#        state.pause_timing
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    mod = M().to(device).to(data_type)
    opt_mod = torch._dynamo.optimize("inductor")(mod)
    memory_format = torch.channels_last
    inputs = (
        torch.randn([N, C_in, H, W]).to(data_type).to(memory_format=memory_format).to(device),
    )
    # y = mod(*inputs)
    # state.resume_timing
    # milliseconds
    start_time_ms = time.time_ns() // 1_000_000
    opt_y = opt_mod(*inputs)
    elapsed_time_ms = (time.time_ns() // 1_000_000) - start_time_ms
    return elapsed_time_ms
    #    self.assertEqual(y, opt_y)

data_types = {"f32" : torch.float32,
              "f16" : torch.float16,}

assert(torch.cuda.is_available())
assert(has_triton())

with open('../benchmark_sizes/resnet50.json') as f:
    # warmup
# test_conv_extern_kernel(1, 16, 33, 50, 100, 2, 2, (1,1), (0,0), (1,1))
#    data = json.load(f)
    data = load_csv('../benchmark_sizes/sd.csv')
    data_type = args.precision
    dtype = data_types[data_type]
#    state = benchmark.State
    times = []
    labels = []
    for config in data['configs']:
        input, filter, output, strides, padding, dilation = get_config(config)
#        benchmark.register(test_conv_extern_kernel(state, N=input['N'], C_in=input['C'], C_out=filter['F'],
#                                                   H=input['H'], W=input['W'],
#                                                   kW=filter['H'], kH=filter['W'],
#                                                   strides=tuple(strides), padding=tuple(padding),
#                                                   dilations=tuple(dilation)))
# benchmark.main()
        t = test_conv_extern_kernel(N=input['N'], C_in=input['C'], C_out=filter['F'], \
                                    H=input['H'], W=input['W'], \
                                    kW=filter['H'], kH=filter['W'], \
                                    strides=strides, padding=padding, dilations=dilation, data_type=dtype)

        label = get_labels(input, filter, output)
        label['strides'] = strides
        labels.append(label)
        times.append(t)

header = 'op, input_shape, filter_shape, output_shape, stride, dtype, ms'
print(header)
for label, t in zip(labels, times):
    line = '{}, {}, {}, {}, {}, {}, {}'.format(label['op'], label['input_shape'], label['filter_shape'], \
            label['output_shape'], label['strides'], data_type, t)
    print(line)


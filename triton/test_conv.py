import torch
import triton
import torch._dynamo
from torch._inductor import config

import argparse
from convperf.python.reader import load_csv, get_config, get_labels

p = argparse.ArgumentParser()

p.add_argument(
    "--precision",
    type=str,
    default="f32",
    help="f16 or f32",
)
args = p.parse_args()

@config.patch({"triton.convolution" : "triton"})
def run_conv(n, c_in, h, w, c_out, kW, kH, stride, padding, dilation, d_type):
    class M(torch.nn.Module):
        def __init__(
            self,
            **kwargs,
        ):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(
                             c_in,
                             c_out,
                             (kH,kW),
                             stride,
                             padding,
                             dilation,
                         )

        def forward(self, x):
            x1 = self.conv1(x)
            return x1

    mod = M().cuda().to(d_type)

    opt_mod = torch._dynamo.optimize("inductor")(mod)
    inputs = (
        torch.randn([n, c_in, h, w], dtype=d_type, device="cuda"),
    )
    opt_y = opt_mod(*inputs)
    ms, min_ms, max_ms = triton.testing.do_bench(
                             lambda: opt_mod(*inputs),
                             rep=100)
    return ms

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
                    kH=filter['H'], kW=filter['W'], stride=(strides, strides),
                    padding=(padding, padding), dilation=(dilation, dilation),
                    d_type=dtype)
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


# print(run_conv())

export PYTHONPATH=[/path/to/AITemplate/python/]:/dir/containing/convperf/

install pytorch from source, USE_CUDA=1, CUDACXX
pip install triton==2.0.0.dev20221030

has_triton() is True


python test_conv.py [--precision=f16|f32]

nsys profile:
/usr/local/cuda/bin/nsys profile --trace=cuda,cudnn,nvtx,opengl -o my_test python3 conv.py

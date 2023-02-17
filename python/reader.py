import re
import csv

def split_shape(shape):
    return [int(i) for i in shape.split('x')]

def map_shape(keys, shape):
    shape_dict = {}
    for i in range(len(keys)):
        shape_dict[keys[i]] = shape[i]
    return shape_dict

def split_str(s):
    return [letter for letter in s]

def load_csv(fn):
    data = {'configs' : []}

    with open(fn, "r+") as f:
        reader = csv.reader(f, delimiter=",")
        reader.__next__()
        for row in reader:
            print(row)
            op = row[0]
            input_shape = row[1]
            filter_shape = row[2]
            output_shape = row[3]
            stride = row[4]

            dims = re.match(r'linalg\.conv_2d_(\w+)_(\w+)', op)
            input_dims = dims.group(1)
            filter_dims = dims.group(2)
            input = map_shape(split_str(input_dims.upper()), split_shape(input_shape))
            input['format'] = input_dims
            filter = map_shape(split_str(filter_dims.upper()), split_shape(filter_shape))
            filter['format'] = filter_dims
            output = map_shape(split_str(input_dims.upper()), split_shape(output_shape))
            output['format'] = input_dims
            config = {'input' : input, 'filter' : filter, 'output' : output, 'strides' : int(stride)}
            data['configs'].append(config)
    return data

def get_config(config):
    input = config['input']
    filter = config['filter']
    output = config['output']
    if type(config['strides']) is not int:
        strides = tuple(config['strides'])
    else:
        strides = config['strides']
    if 'padding' in config:
        padding = tuple(config['padding'])
    else:
        padding = 0
    if 'dilation' in config:
        dilation = tuple(config['dilations'])
    else:
        dilation = 1

    return input, filter, output, strides, padding, dilation

def get_labels(input, filter, output):
    input_list = []
    filter_list = []
    output_list = []
    dims_dict = {}
    for ch in input['format']:
        if ch == 'n':
            input_list.append(str(input['N']))
        if ch == 'c':
            input_list.append(str(input['C']))
        if ch == 'h':
            input_list.append(str(input['H']))
        if ch == 'w':
            input_list.append(str(input['W']))
    for ch in output['format']:
        if ch == 'n':
            output_list.append(str(output['N']))
        if ch == 'c':
            output_list.append(str(output['C']))
        if ch == 'h':
            output_list.append(str(output['H']))
        if ch == 'w':
            output_list.append(str(output['W']))
    for ch in filter['format']:
        if ch == 'f':
            filter_list.append(str(filter['F']))
        if ch == 'c':
            filter_list.append(str(filter['C']))
        if ch == 'h':
            filter_list.append(str(filter['H']))
        if ch == 'w':
            filter_list.append(str(filter['W']))

    dims_dict['input'] = 'x'.join(input_list)
    dims_dict['output'] = 'x'.join(output_list)
    dims_dict['filter'] = 'x'.join(filter_list)
    labels = {'op' : 'conv2d_{}_{}'.format(input['format'], filter['format']), \
              'output_shape' : dims_dict['output'], \
              'input_shape' : dims_dict['input'], 'filter_shape' : dims_dict['filter']}

    return labels

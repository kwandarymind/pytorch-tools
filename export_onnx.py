import torch
import torch.onnx
import os, argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser() 
    parser.add_argument('-o', '--output', help='Output onnx file name', default='model.onnx')
    requiredNamed = parser.add_argument_group('required named arguments')
    requiredNamed.add_argument('-i', '--input', help='Input model file name', required=True)
    args = parser.parse_args(['-h'])

    dummy_input = torch.randn(1, 3, 513, 513)
    state_dict = torch.load(args.input)
    model.load_state_dict(state_dict)

    torch.onnx.export(model, dummy_input, args.output)
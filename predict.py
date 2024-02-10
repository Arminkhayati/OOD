import argparse
from xray_model import XrayModel

parser = argparse.ArgumentParser(description='Predict an image')
parser.add_argument('-i', '--image', default="", type=str, help='Image file name')
parser.add_argument('-ood', '--ood-method', default=None, type=str, help='Out of distribution detection method')
parser.add_argument('-dd', '--data-dir', default="data/x_ray", type=str, help='Image directory')
parser.add_argument('-m', '--model', default="model/vgg16-0.96-full_model.h5", type=str, help='Model Path')
parser.add_argument('-s', '--strict', default=True, type=bool, help='Run prediction based on stricted or non-stricted threshold.')
parser.add_argument('-rl', '--return_label', default=False, type=bool, help='Whether to return string label or int index of the predicted class.')


args = parser.parse_args()

def main():
    model = XrayModel(model_path=args.model,
                      ood_method=args.ood_method,
                      data_dir=args.data_dir,
                      strict=args.strict,
                      return_label=args.return_label)
    label = model.predict(args.image)
    print()
    print(f"Label: {label}")
    print()

if __name__ == '__main__':
    main()

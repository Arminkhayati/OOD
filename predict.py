import argparse
from xray_model import XrayModel

parser = argparse.ArgumentParser(description='Predict an image')
parser.add_argument('-i', '--image', default="", type=str, help='Image file name')
parser.add_argument('-ood', '--ood-method', default=None, type=str, help='Out of distribution detection method')
parser.add_argument('-dd', '--data-dir', default="data/x_ray", type=str, help='Image directory')
parser.add_argument('-m', '--model', default="model/vgg16-0.96-full_model.h5", type=str, help='Model Path')

args = parser.parse_args()

def main():
    model = XrayModel(model_path=args.model,
                      ood_method=args.ood_method,
                      data_dir=args.data_dir)
    label = model.predict(args.image)
    print()
    print(f"Label: {label}")
    print()

if __name__ == '__main__':
    main()

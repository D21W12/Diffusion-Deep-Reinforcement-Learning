import argparse

from .evaluation import evaluate

def main():

    parser = argparse.ArgumentParser(
        prog='Project evaluation',
        description='This program manages evaluation for my BSc Thesis project.',
        epilog='That are all commands >.<'
    )

    parser.add_argument('-c', '--checkpoint', required=True)
    parser.add_argument('-o', '--output')

    parser.add_argument('-d', '--device', default='cpu')
    parser.add_argument('-m', '--model', required=True)

    parser.add_argument('-n', '--number', type=int)
    parser.add_argument("--manual", action='store_true')

    args = parser.parse_args()

    kwargs = {
        "model": args.model,
        "device": args.device,
        "checkpoint_path": args.checkpoint,
        "manual": args.manual
    }
    if args.output is not None: kwargs["output_path"] = args.output
    if args.number is not None: kwargs["number"] = args.number

    evaluate(**kwargs)

if __name__ == "__main__":
    main()

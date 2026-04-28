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

    parser.add_argument('-e', '--episodes', type=int)
    parser.add_argument("-m", "--manual", action='store_true')

    args = parser.parse_args()

    kwargs = {
        "episodes": args.episodes,
        "device": args.device,
        "checkpoint_path": args.checkpoint,
        "output_path": args.output,
        "manual": args.manual
    }
    if args.episodes: kwargs["episodes"] = args.episodes

    evaluate(**kwargs)

if __name__ == "__main__":
    main()

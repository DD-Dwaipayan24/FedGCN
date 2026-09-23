import argparse
from train_fedgcn import train, plot_metric


DATASETS = (
    "Cora", 
    "Citeseer", 
    "Pubmed", 
    "Actor", 
    "Chameleon", 
    "Cornell", 
    "Texas", 
    "Wisconsin",
    "Roman-empire",
    "amazon-ratings"
)



def parse_args():
    parser = argparse.ArgumentParser(description="Federated GCN (FedGCN) on citation graphs")
    parser.add_argument(
        "--dataset", type=str, default="Cora", choices=DATASETS
        )
    parser.add_argument(
        "--num_clients", type=int, default=10, help="Total number of clients (K)"
        )
    parser.add_argument(
        "--rounds", type=int, default=50, help="Communication rounds (T)"
        )
    parser.add_argument(
        "--frac", type=float, default=0.4, help="Fraction of clients sampled per round (C)"
        )
    parser.add_argument(
        "--local_epochs", type=int, default=3, help="Local full-batch epochs per client (E)"
        )
    parser.add_argument(
        "--lr", type=float, default=0.01, help="Local Adam learning rate"
        )
    parser.add_argument(
        "--weight_decay", type=float, default=5e-4
        )
    parser.add_argument(
        "--hidden-dims", type=int, nargs="+", default=[16],
        help="Sizes of the hidden layers between the input and output layers "
    )
    parser.add_argument(
        "--hops", type=int, default=2, 
        help="Pre-aggregated hops (L); matches the model's effective depth"
        )
    parser.add_argument(
        "--dropout", type=float, default=0.5
        )
    parser.add_argument(
        "--iid", action="store_true", help="Use a uniform random node split (default: community split)"
        )
    parser.add_argument(
        "--seed", type=int, default=42
        )
    parser.add_argument(
        "--data_dir", type=str, default="./data"
        )
    parser.add_argument(
        "--out", type=str, default="fedgcn_model.pth", help="Filename (in outputs/) for the final global model"
        )
    parser.add_argument(
        "--dpi", type=int, default=1000,
        help="DPI used when saving plot images (default: %(default)s)",
    )
    parser.add_argument(
        "--no-show", action="store_true",
        help="Save plots without opening an interactive window",
    )
    parser.add_argument(
        "--log-every", type=int, default=1,
        help="Print training stats every N epochs (default: %(default)s)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    history = train(args)

    plot_metric(
        args,
        history,
        args.rounds,
        args.dataset,
        args.dpi,
        args.show
    )

if __name__ == "__main__":
    main()
from torch_geometric.datasets import Planetoid, Actor, WikipediaNetwork, WebKB, HeterophilousGraphDataset

def load_dataset(name: str, root: str):
    if name == "Cora" or name =="Citeseer" or name == "Pubmed":
        return Planetoid(root=f"{root}/{name}", name=name)
    if name == "Actor":
        return Actor(root = f"{root}/{name}")
    if name == "Chameleon":
        return WikipediaNetwork(root = f"{root}/{name}", name = "chameleon")
    if name == "Cornell" or name == "Texas" or name == "Wisconsin":
        return WebKB(root = f"{root}/{name}", name = name)
    if name == "Roman-empire" or name == "amazon-ratings":
        return HeterophilousGraphDataset(root=f"{root}/{name}", name=name)

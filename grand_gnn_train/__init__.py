"""Training of the GNN reconstruction networks of grand.analysis.gnn.

``python -m grand_gnn_train --gnn-model-card gnn_model_cards/<name>.json``: the GNN
model card defines the network; its dataset is built (or reused), its seeds trained,
and the network exported for grand.  The graph inputs, the features and the
network are imported from grand, so a network is trained on exactly what
grand gives it at inference.
"""

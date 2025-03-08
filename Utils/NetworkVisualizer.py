import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from itertools import product

def create_edges_batch(nodes_from, nodes_to):
    return [(n1, n2) for n1, n2 in product(nodes_from, nodes_to)]

def calc_pos(i, total, layer_idx=0, layer_spacing=2):
    return (layer_idx * layer_spacing, (i - total/2) * 2)

def generate_network_graph(network):
    """
    Visualizes the structure of a neural network using networkx and matplotlib.
    It just works at small sized networks.
    
    Args:
        network: Instance of NeuralNetwork containing the network structure
    """
    # Create directed graph
    graph = nx.DiGraph()

    # Create node names
    input_nodes = [f'I{i+1}' for i in range(network.input_size)]
    hidden_layers_nodes = [[f'H{i+1}_{j+1}' for j in range(n)] for i, n in enumerate(network.hidden_layers)]
    output_nodes = [f'O{i+1}' for i in range(network.output_size)]

    # Define node positions
    pos = {}
    layer_spacing = 2
    
    # Input layer positions
    for i, node in enumerate(input_nodes):
        pos[node] = np.array(calc_pos(i, len(input_nodes)))

    # Hidden layers positions
    for layer_idx, layer in enumerate(hidden_layers_nodes):
        for i, node in enumerate(layer):
            pos[node] = np.array(calc_pos(i, len(layer), layer_idx + 1))

    # Output layer positions
    for i, node in enumerate(output_nodes):
        pos[node] = np.array(calc_pos(i, len(output_nodes), len(hidden_layers_nodes) + 1))

    # Add nodes to graph
    graph.add_nodes_from(input_nodes)
    for layer in hidden_layers_nodes:
        graph.add_nodes_from(layer)
    graph.add_nodes_from(output_nodes)

    # Create connections
    # Input → first hidden layer
    edges = create_edges_batch(input_nodes, hidden_layers_nodes[0])
    graph.add_edges_from(edges)

    # Between hidden layers
    for i in range(len(network.hidden_layers) - 1):
        edges = create_edges_batch(hidden_layers_nodes[i], hidden_layers_nodes[i+1])
        graph.add_edges_from(edges)

    # Last hidden layer → output
    edges = create_edges_batch(hidden_layers_nodes[-1], output_nodes)
    graph.add_edges_from(edges)

    # Create figure
    plt.figure(figsize=(12, 8))

    # Draw graph
    nx.draw(graph, pos, with_labels=True, 
           node_color='lightblue',
           node_size=1000,
           arrowsize=20,
           font_size=10,
           font_weight='bold')

    # Configure visualization
    plt.title('Neural Network Structure')
    plt.axis('off')
    
    # Show graph
    plt.show()

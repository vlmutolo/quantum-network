import networkx as nx
import random



class DirectGraph:
    def __init__(self):
        """Initialize an empty undirected graph."""
        self.G = nx.Graph()

    def create_wraparound_grid(self, rows, cols, seed=None):
        """
        Creates a wraparound grid (torus topology) using NetworkX Graph.

        Args:
            rows (int): Number of rows in the grid
            cols (int): Number of columns in the grid
            seed (int, optional): Random seed for reproducible results

        Returns:
            self: Returns self for method chaining
        """
        if seed is not None:
            random.seed(seed)

        # Clear existing graph
        self.G.clear()

        # Add nodes
        for i in range(rows):
            for j in range(cols):
                self.G.add_node((i, j))

        # Add undirected edges with wraparound connectivity
        for i in range(rows):
            for j in range(cols):
                current_node = (i, j)

                # Right neighbor (wrap around horizontally)
                right_neighbor = (i, (j + 1) % cols)
                if not self.G.has_edge(current_node, right_neighbor):
                    c_rate = random.random()  # Random value in [0, 1]
                    self.G.add_edge(current_node, right_neighbor, c=c_rate)

                # Bottom neighbor (wrap around vertically)
                bottom_neighbor = ((i + 1) % rows, j)
                if not self.G.has_edge(current_node, bottom_neighbor):
                    c_rate = random.random()  # Random value in [0, 1]
                    self.G.add_edge(current_node, bottom_neighbor, c=c_rate)

        return self

    def get_generation_rate(self, node1, node2):
        """
        Get the generation rate between node1 and node2.

        Args:
            node1: First node (i, j) tuple
            node2: Second node (i, j) tuple

        Returns:
            float: Generation rate c(i,j) if edge exists, None otherwise
        """
        if self.G.has_edge(node1, node2):
            return self.G[node1][node2]['c']
        return None

    def set_generation_rate(self, node1, node2, rate):
        """
        Set the generation rate between node1 and node2.

        Args:
            node1: First node (i, j) tuple
            node2: Second node (i, j) tuple
            rate (float): Generation rate to set (should be positive)
        """
        if self.G.has_edge(node1, node2) and rate > 0:
            self.G[node1][node2]['c'] = rate

    def get_graph(self):
        """
        Get the underlying NetworkX Graph object.

        Returns:
            networkx.Graph: The undirected graph
        """
        return self.G

    def get_nodes(self):
        """
        Get all nodes in the graph.

        Returns:
            NodeView: All nodes in the graph
        """
        return self.G.nodes()

    def get_edges(self):
        """
        Get all edges in the graph with their data.

        Returns:
            EdgeDataView: All edges with their attributes
        """
        return self.G.edges(data=True)

    def has_edge(self, node1, node2):
        """
        Check if there's an edge between node1 and node2.

        Args:
            node1: First node
            node2: Second node

        Returns:
            bool: True if edge exists, False otherwise
        """
        return self.G.has_edge(node1, node2)

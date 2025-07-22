#include <iostream>
#include <vector>
#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <random>
#include <cmath>
#include <queue>
#include <fstream>
#include <string>
#include <iterator>
#include <numeric>

using namespace std;

// Parameters
const int n = 50;
const int initial_capacity = 20;
const int generation_rate = 40;
const int edges_with_consumption_non_zero_rate = 50;

// Pair hash function for using pairs as keys in unordered_map
struct PairHash {
    size_t operator()(const pair<int, int>& p) const {
        return hash<int>()(p.first) ^ (hash<int>()(p.second) << 1);
    }
};

// Node type for grid graphs
using Node = pair<int, int>;

// Random number generator
random_device rd;
mt19937 gen(rd());

pair<int, int> norm_edge(int u, int v) {
    return u <= v ? make_pair(u, v) : make_pair(v, u);
}

pair<Node, Node> norm_edge(Node u, Node v) {
    return u <= v ? make_pair(u, v) : make_pair(v, u);
}

vector<int> generate_recursive_array(int D, int N = 50) {
    vector<int> f(N, 0);
    if (N > 1) {
        f[1] = D;
    }
    for (int n = 2; n < N; n++) {
        f[n] = D * (f[n / 2] + f[(n + 1) / 2]);
    }
    return f;
}

// Graph class to handle both integer and Node types
template<typename NodeType>
class Graph {
public:
    map<pair<NodeType, NodeType>, int> edges;
    set<NodeType> nodes;
    
    void add_node(NodeType node) {
        nodes.insert(node);
    }
    
    void add_edge(NodeType u, NodeType v, int capacity) {
        auto edge = u <= v ? make_pair(u, v) : make_pair(v, u);
        edges[edge] = capacity;
        nodes.insert(u);
        nodes.insert(v);
    }
    
    bool has_edge(NodeType u, NodeType v) {
        auto edge = u <= v ? make_pair(u, v) : make_pair(v, u);
        return edges.find(edge) != edges.end();
    }
    
    int number_of_edges() {
        return edges.size();
    }
    
    vector<NodeType> get_nodes() const {
        return vector<NodeType>(nodes.begin(), nodes.end());
    }
    
    vector<pair<NodeType, NodeType>> get_edges() const {
        vector<pair<NodeType, NodeType>> edge_list;
        for (const auto& e : edges) {
            edge_list.push_back(e.first);
        }
        return edge_list;
    }
    
    // BFS to check connectivity for grid graphs
    bool is_connected() {
        if (nodes.empty()) return true;
        
        set<NodeType> visited;
        queue<NodeType> q;
        
        NodeType start = *nodes.begin();
        q.push(start);
        visited.insert(start);
        
        while (!q.empty()) {
            NodeType current = q.front();
            q.pop();
            
            for (auto& edge : edges) {
                NodeType neighbor;
                if (edge.first.first == current) {
                    neighbor = edge.first.second;
                } else if (edge.first.second == current) {
                    neighbor = edge.first.first;
                } else {
                    continue;
                }
                
                if (visited.find(neighbor) == visited.end()) {
                    visited.insert(neighbor);
                    q.push(neighbor);
                }
            }
        }
        
        return visited.size() == nodes.size();
    }
};

Graph<Node> create_overlay_on_grid_graph(int initial_capacity, bool wrap = true) {
    Graph<Node> G;
    int rows = 10, cols = 5;
    
    // Add nodes
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            G.add_node(make_pair(i, j));
        }
    }
    
    // Generate all possible edges
    vector<pair<Node, Node>> edges;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            Node current = make_pair(i, j);
            Node right = wrap ? make_pair(i, (j + 1) % cols) : make_pair(i, j + 1);
            Node down = wrap ? make_pair((i + 1) % rows, j) : make_pair(i + 1, j);
            
            if (wrap || j + 1 < cols) {
                edges.push_back(make_pair(current, right));
            }
            if (wrap || i + 1 < rows) {
                edges.push_back(make_pair(current, down));
            }
        }
    }
    
    // Shuffle edges
    shuffle(edges.begin(), edges.end(), gen);
    
    // Add edges until connected
    for (auto& edge : edges) {
        G.add_edge(edge.first, edge.second, initial_capacity);
        if (G.is_connected()) {
            break;
        }
    }
    
    // Add extra edges (20% of total)
    int extra_edges_to_add = static_cast<int>(0.2 * edges.size());
    int added = 0;
    for (auto& edge : edges) {
        if (!G.has_edge(edge.first, edge.second)) {
            G.add_edge(edge.first, edge.second, initial_capacity);
            added++;
        }
        if (added >= extra_edges_to_add) {
            break;
        }
    }
    
    return G;
}

// BFS shortest path for grid graphs
map<pair<Node, Node>, int> compute_shortest_paths(const Graph<Node>& G) {
    map<pair<Node, Node>, int> distances;
    vector<Node> nodes = G.get_nodes();
    
    for (Node start : nodes) {
        map<Node, int> dist;
        queue<Node> q;
        q.push(start);
        dist[start] = 0;
        
        while (!q.empty()) {
            Node current = q.front();
            q.pop();
            
            for (auto& edge : G.edges) {
                Node neighbor;
                if (edge.first.first == current) {
                    neighbor = edge.first.second;
                } else if (edge.first.second == current) {
                    neighbor = edge.first.first;
                } else {
                    continue;
                }
                
                if (dist.find(neighbor) == dist.end()) {
                    dist[neighbor] = dist[current] + 1;
                    q.push(neighbor);
                }
            }
        }
        
        for (auto& d : dist) {
            distances[make_pair(start, d.first)] = d.second;
        }
    }
    
    return distances;
}

// Simple shortest path for cycle/line graphs
map<pair<int, int>, int> compute_shortest_paths_simple(int n, const string& graph_type) {
    map<pair<int, int>, int> distances;
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (graph_type == "cycle") {
                distances[make_pair(i, j)] = min(abs(i - j), n - abs(i - j));
            } else { // line
                distances[make_pair(i, j)] = abs(i - j);
            }
        }
    }
    
    return distances;
}

pair<Graph<int>, map<pair<int, int>, int>> create_simple_graph(int n, int initial_capacity, const string& graph_type) {
    Graph<int> G;
    
    if (graph_type == "cycle") {
        for (int i = 0; i < n; i++) {
            G.add_edge(i, (i + 1) % n, initial_capacity);
        }
    } else if (graph_type == "line") {
        for (int i = 0; i < n - 1; i++) {
            G.add_edge(i, i + 1, initial_capacity);
        }
    }
    
    // Generate random consumption edges
    vector<pair<int, int>> all_possible_edges;
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            all_possible_edges.push_back(make_pair(i, j));
        }
    }
    
    shuffle(all_possible_edges.begin(), all_possible_edges.end(), gen);
    
    map<pair<int, int>, int> consumption_rates;
    uniform_int_distribution<int> rate_dist(1, 3);
    
    for (int i = 0; i < min(edges_with_consumption_non_zero_rate, (int)all_possible_edges.size()); i++) {
        consumption_rates[all_possible_edges[i]] = rate_dist(gen);
    }
    
    return make_pair(G, consumption_rates);
}

pair<Graph<Node>, map<pair<Node, Node>, int>> create_grid_graph(int initial_capacity, const string& graph_type) {
    bool wrap = (graph_type.find("wrap") != string::npos);
    Graph<Node> G = create_overlay_on_grid_graph(initial_capacity, wrap);
    
    // Generate random consumption edges for grid
    vector<Node> nodes = G.get_nodes();
    vector<pair<Node, Node>> all_possible_edges;
    
    for (size_t i = 0; i < nodes.size(); i++) {
        for (size_t j = i + 1; j < nodes.size(); j++) {
            auto edge = nodes[i] <= nodes[j] ? make_pair(nodes[i], nodes[j]) : make_pair(nodes[j], nodes[i]);
            all_possible_edges.push_back(edge);
        }
    }
    
    shuffle(all_possible_edges.begin(), all_possible_edges.end(), gen);
    
    map<pair<Node, Node>, int> consumption_rates;
    uniform_int_distribution<int> rate_dist(1, 3);
    
    for (int i = 0; i < min(edges_with_consumption_non_zero_rate, (int)all_possible_edges.size()); i++) {
        consumption_rates[all_possible_edges[i]] = rate_dist(gen);
    }
    
    return make_pair(G, consumption_rates);
}

bool is_stabilized(const vector<double>& arr, int num_last = 20, double tolerance = 0.005) {
    if ((int)arr.size() < max(2, num_last)) {
        return false;
    }
    
    vector<double> diffs;
    for (int i = arr.size() - num_last; i < (int)arr.size() - 1; i++) {
        diffs.push_back(abs(arr[i + 1] - arr[i]));
    }
    
    return *max_element(diffs.begin(), diffs.end()) < tolerance;
}

template<typename NodeType>
set<NodeType> get_neighbours(const map<pair<NodeType, NodeType>, int>& edge_states, NodeType x) {
    set<NodeType> neighbors;
    for (auto& edge : edge_states) {
        if (edge.second > 0) {
            if (edge.first.first == x) {
                neighbors.insert(edge.first.second);
            } else if (edge.first.second == x) {
                neighbors.insert(edge.first.first);
            }
        }
    }
    return neighbors;
}

template<typename NodeType>
vector<pair<pair<NodeType, NodeType>, int>> get_preferable_swaps(
    const set<NodeType>& neighbors, 
    const map<pair<NodeType, NodeType>, int>& edge_states, 
    NodeType x, 
    int distillation_pairs) {
    
    vector<pair<pair<NodeType, NodeType>, int>> swaps;
    vector<NodeType> neighbor_vec(neighbors.begin(), neighbors.end());
    
    for (size_t i = 0; i < neighbor_vec.size(); i++) {
        for (size_t j = i + 1; j < neighbor_vec.size(); j++) {
            NodeType y = neighbor_vec[i];
            NodeType z = neighbor_vec[j];
            
            auto edge_xy = x <= y ? make_pair(x, y) : make_pair(y, x);
            auto edge_xz = x <= z ? make_pair(x, z) : make_pair(z, x);
            auto edge_yz = y <= z ? make_pair(y, z) : make_pair(z, y);
            
            int c_xy = edge_states.count(edge_xy) ? edge_states.at(edge_xy) : 0;
            int c_xz = edge_states.count(edge_xz) ? edge_states.at(edge_xz) : 0;
            int c_yz = edge_states.count(edge_yz) ? edge_states.at(edge_yz) : 0;
            
            if (c_xy >= distillation_pairs && c_xz >= distillation_pairs && 
                c_xy > c_yz + distillation_pairs && c_xz > c_yz + distillation_pairs) {
                swaps.push_back(make_pair(make_pair(y, z), c_yz));
            }
        }
    }
    
    return swaps;
}

template<typename NodeType>
tuple<bool, int, int> attempt_preferable_swap(
    NodeType x,
    const set<NodeType>& neighbors,
    map<pair<NodeType, NodeType>, int>& edge_states,
    int distillation_pairs,
    int swap_count,
    int total_bell_pairs) {
    
    auto swaps = get_preferable_swaps(neighbors, edge_states, x, distillation_pairs);
    
    // Sort by c_yz (ascending)
    sort(swaps.begin(), swaps.end(), [](const auto& a, const auto& b) {
        return a.second < b.second;
    });
    
    for (auto& swap : swaps) {
        NodeType y = swap.first.first;
        NodeType z = swap.first.second;
        
        auto edge_xy = x <= y ? make_pair(x, y) : make_pair(y, x);
        auto edge_xz = x <= z ? make_pair(x, z) : make_pair(z, x);
        auto edge_yz = y <= z ? make_pair(y, z) : make_pair(z, y);
        
        if (edge_states[edge_xy] >= distillation_pairs && edge_states[edge_xz] >= distillation_pairs) {
            edge_states[edge_xy] -= distillation_pairs;
            edge_states[edge_xz] -= distillation_pairs;
            if (edge_states.find(edge_yz) == edge_states.end()) {
                edge_states[edge_yz] = 0;
            }
            edge_states[edge_yz] += 1;
            return make_tuple(true, swap_count + 1, total_bell_pairs - 2 * distillation_pairs + 1);
        }
    }
    
    return make_tuple(false, swap_count, total_bell_pairs);
}

template<typename NodeType>
tuple<int, int, int, int, int, int> simulate(
    Graph<NodeType>& G,
    map<pair<NodeType, NodeType>, int>& consumption_rates,
    int time_steps,
    int generation_rate,
    int swap_rate,
    int distillation_pairs,
    bool infinite_swapping,
    const map<pair<NodeType, NodeType>, int>& shortest_paths) {
    
    map<pair<NodeType, NodeType>, int> edge_states;
    for (auto& edge : G.edges) {
        edge_states[edge.first] = edge.second;
    }
    
    vector<NodeType> nodes = G.get_nodes();
    vector<int> distillation_sum_multiplier = generate_recursive_array(distillation_pairs);
    
    vector<pair<NodeType, NodeType>> consumption_edges;
    vector<int> consumption_weights;
    for (auto& cr : consumption_rates) {
        consumption_edges.push_back(cr.first);
        consumption_weights.push_back(cr.second);
    }
    
    int total_swap_count = 0;
    int failed = 0;
    int succeeded = 0;
    int total_bell_pairs = initial_capacity * G.number_of_edges();
    int optimal_swap_count = 0;
    vector<double> swap_overlay;
    
    // Setup distributions
    vector<int> event_weights = {
        (int)G.get_edges().size() * generation_rate,
        accumulate(consumption_weights.begin(), consumption_weights.end(), 0),
        swap_rate * (int)nodes.size()
    };
    
    discrete_distribution<int> event_dist(event_weights.begin(), event_weights.end());
    uniform_int_distribution<int> edge_dist(0, G.get_edges().size() - 1);
    discrete_distribution<int> consumption_dist(consumption_weights.begin(), consumption_weights.end());
    uniform_int_distribution<int> node_dist(0, nodes.size() - 1);
    
    for (int t = 1; t <= time_steps; t++) {
        int event = event_dist(gen);
        
        if (event == 0) { // generate
            auto edges = G.get_edges();
            auto edge = edges[edge_dist(gen)];
            auto norm_edge_key = edge.first <= edge.second ? edge : make_pair(edge.second, edge.first);
            edge_states[norm_edge_key]++;
            total_bell_pairs++;
        } else if (event == 1) { // consume
            int idx = consumption_dist(gen);
            auto edge = consumption_edges[idx];
            auto norm_edge_key = edge.first <= edge.second ? edge : make_pair(edge.second, edge.first);
            
            if (edge_states.count(norm_edge_key) && edge_states[norm_edge_key] >= distillation_pairs) {
                edge_states[norm_edge_key] -= distillation_pairs;
                auto path_key = make_pair(edge.first, edge.second);
                int path_length = shortest_paths.at(path_key);
                optimal_swap_count += distillation_sum_multiplier[path_length - 1];
                succeeded++;
                total_bell_pairs -= distillation_pairs;
            } else {
                failed++;
            }
        } else { // swap
            if (!infinite_swapping) {
                NodeType x = nodes[node_dist(gen)];
                set<NodeType> neighbors = get_neighbours(edge_states, x);
                auto result = attempt_preferable_swap(x, neighbors, edge_states, distillation_pairs, total_swap_count, total_bell_pairs);
                total_swap_count = get<1>(result);
                total_bell_pairs = get<2>(result);
            } else {
                vector<NodeType> shuffled_nodes = nodes;
                shuffle(shuffled_nodes.begin(), shuffled_nodes.end(), gen);
                
                for (NodeType x : shuffled_nodes) {
                    set<NodeType> neighbors = get_neighbours(edge_states, x);
                    auto result = attempt_preferable_swap(x, neighbors, edge_states, distillation_pairs, total_swap_count, total_bell_pairs);
                    bool changed = get<0>(result);
                    total_swap_count = get<1>(result);
                    total_bell_pairs = get<2>(result);
                    if (changed) {
                        break;
                    }
                }
            }
        }
        
        if (t % 20000 == 0) {
            if (t % 100000 == 0) {
                cout << t << " ";
                if (swap_overlay.size() >= 5) {
                    for (int i = swap_overlay.size() - 5; i < (int)swap_overlay.size(); i++) {
                        cout << swap_overlay[i] << " ";
                    }
                }
                cout << endl;
            }
            if (optimal_swap_count > 0) {
                swap_overlay.push_back((double)total_swap_count / optimal_swap_count);
                if (is_stabilized(swap_overlay)) {
                    break;
                }
            }
        }
    }
    
    int correction = 0;
    for (auto& edge : edge_states) {
        if (edge.second >= distillation_pairs) {
            NodeType u = edge.first.first;
            NodeType v = edge.first.second;
            auto path_key = make_pair(u, v);
            correction += distillation_sum_multiplier[shortest_paths.at(path_key) - 1];
        }
    }
    
    return make_tuple(total_swap_count, optimal_swap_count, optimal_swap_count + correction, failed, succeeded, total_bell_pairs);
}

void save_csv(const string& filename, const vector<vector<string>>& data) {
    ofstream file(filename);
    for (size_t i = 0; i < data.size(); i++) {
        for (size_t j = 0; j < data[i].size(); j++) {
            file << data[i][j];
            if (j < data[i].size() - 1) file << ",";
        }
        file << "\n";
    }
    file.close();
}

int main() {
    vector<string> graph_types = {"cycle"};
    
    for (const string& graph_type : graph_types) {
        cout << "Simulating swap overhead for " << graph_type << endl;
        
        // Create graph and get shortest paths
        auto graph_data = create_simple_graph(n, initial_capacity, graph_type);
        auto G = graph_data.first;
        auto consumption_rates = graph_data.second;
        auto shortest_paths = compute_shortest_paths_simple(n, graph_type);
        
        vector<int> swap_rates;
        for (int rate = 150; rate < 190; rate += 50) {
            swap_rates.push_back(rate);
        }
        
        vector<vector<string>> csv_data;
        csv_data.push_back({"swap rate", "total swaps", "optimal", "corrected", "failed", "succeeded", "final bell pairs"});
        
        for (int rate : swap_rates) {
            auto result = simulate(G, consumption_rates, 2000000, generation_rate, rate, 1, false, shortest_paths);
            csv_data.push_back({
                to_string(rate),
                to_string(get<0>(result)),
                to_string(get<1>(result)),
                to_string(get<2>(result)),
                to_string(get<3>(result)),
                to_string(get<4>(result)),
                to_string(get<5>(result))
            });
        }
        
        save_csv(graph_type + "_swap_rate_metrics_output.csv", csv_data);
        
        cout << "Simulating distillation effect for " << graph_type << endl;
        
        vector<int> distillations;
        for (int d = 1; d < 8; d++) {
            distillations.push_back(d);
        }
        
        csv_data.clear();
        csv_data.push_back({"distillation pairs", "total swaps", "optimal", "corrected", "failed", "succeeded", "final bell pairs"});
        
        for (int d : distillations) {
            auto result = simulate(G, consumption_rates, 2000000, generation_rate, 300, d, false, shortest_paths);
            csv_data.push_back({
                to_string(d),
                to_string(get<0>(result)),
                to_string(get<1>(result)),
                to_string(get<2>(result)),
                to_string(get<3>(result)),
                to_string(get<4>(result)),
                to_string(get<5>(result))
            });
        }
        
        save_csv(graph_type + "_distillation_pair_cpp_output.csv", csv_data);
    }
    
    return 0;
}
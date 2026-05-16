// ============================================================
// BookSim2 → PyBind11 Python Module
// ============================================================
// File: booksim_pybind.cpp
// Compile: c++ -O3 -shared -fPIC `python3 -m pybind11 --includes`
//          booksim_pybind.cpp routefunc.o ... -o booksim`python3 -m pybind11 --includes`
//          -I. -Iarbiters -Iallocators -Irouters -Inetworks -Ipower
//
// Usage in Python:
//   import booksim
//   env = booksim.NoCEnv(traffic='hotspot', inj_rate=0.1)
//   obs = env.reset()
//   for step in range(100):
//       action = agent(obs)  # DRL agent
//       obs, reward, done = env.step(action)
// ============================================================

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <string>
#include <vector>

namespace py = pybind11;

// Forward declarations for BookSim2 internals
extern int gNumVCs;
extern int gN;
void InitializeRoutingMap(const Configuration &cfg);
class Configuration;
class Router;
class Flit;
class OutputSet;

// ============================================================
// BookSim2 Simulation Environment (wrapped for Python)
// ============================================================
class BookSimEnv {
private:
    Configuration* config;
    Network* network;
    TrafficManager* traffic_manager;
    
    int k;              // mesh dimension (e.g., 4 for 4x4)
    int num_nodes;
    int cycle_count;
    int update_period;  // P cycles between AI updates
    
    // Current state matrix [num_nodes x 5]
    // Features: buffer_occ, injection_rate, congestion, vc_util, crossbar
    std::vector<std::vector<float>> state;
    
    // Current routing table [num_nodes x num_nodes]
    // 0=XY, 1=YX
    std::vector<std::vector<int>> routing_table;
    
public:
    BookSimEnv(int k_ = 4, std::string traffic = "hotspot", 
               float inj_rate = 0.1, int period = 5000)
        : k(k_), num_nodes(k_ * k_), cycle_count(0), update_period(period) {
        
        // 1. Create BookSim2 configuration
        config = new Configuration();
        config->AddStrField("topology", "mesh");
        config->AddIntField("k", k);
        config->AddIntField("n", 2);
        config->AddStrField("routing_function", "gnn_ppo_route_4x4");
        config->AddIntField("num_vcs", 4);
        config->AddIntField("vc_buf_size", 8);
        config->AddStrField("traffic", traffic.c_str());
        config->AddFloatField("injection_rate", inj_rate);
        config->AddStrField("sim_type", "latency");
        config->AddIntField("warmup_periods", 3);
        config->AddIntField("sample_period", 1000);
        config->AddIntField("sim_count", 1);
        config->AddIntField("packet_size", 1);
        
        // 2. Initialize routing and network
        InitializeRoutingMap(*config);
        
        // 3. Create network
        network = Network::New(*config, "");
        
        // 4. Create traffic manager
        traffic_manager = new TrafficManager(*config, network, NULL);
        
        // 5. Allocate state and routing table
        state.resize(num_nodes, std::vector<float>(5, 0.0f));
        routing_table.resize(num_nodes, std::vector<int>(num_nodes, 0));
    }
    
    ~BookSimEnv() {
        delete traffic_manager;
        delete network;
        delete config;
    }
    
    /// === FUNCTION 1: Get current network state ===
    py::array_t<float> get_network_state() {
        // Extract buffer occupancy, congestion from each router
        auto result = py::array_t<float>({num_nodes, 5});
        auto buf = result.mutable_unchecked<2>();
        
        for (int i = 0; i < num_nodes; i++) {
            const Router* r = network->GetRouter(i);
            if (r) {
                // Buffer occupancy: average across input ports
                float buf_occ = 0.0f;
                for (int p = 0; p < 4; p++) {
                    buf_occ += r->GetUsedCredit(p);
                }
                buf(i, 0) = buf_occ / 4.0f;
                
                // Congestion level
                buf(i, 1) = buf_occ / 8.0f;
                
                // VC utilization
                buf(i, 2) = r->GetUsedCreditForClass(0, 0) / 8.0f;
                
                // Crossbar contention (estimated)
                buf(i, 3) = (r->GetUsedCredit(2) + r->GetUsedCredit(3)) / 16.0f;
                
                // Position encoding
                int g = k;
                buf(i, 4) = (float)(i % g + i / g) / (2.0f * g);
            }
        }
        return result;
    }
    
    /// === FUNCTION 2: Set routing table from AI ===
    void set_routing_table(py::array_t<int> table) {
        auto buf = table.unchecked<2>();
        for (int i = 0; i < num_nodes; i++) {
            for (int j = 0; j < num_nodes; j++) {
                routing_table[i][j] = buf(i, j);
            }
        }
        // The routing table is read by the custom routing function
        // (gnn_ppo_route_4x4_mesh in routefunc.cpp)
    }
    
    /// === FUNCTION 3: Run P simulation cycles ===
    float run_p_cycles(int p) {
        float total_latency = 0.0f;
        int total_packets = 0;
        
        for (int c = 0; c < p; c++) {
            traffic_manager->_Run();  // Run one cycle
            
            // Collect stats
            // (simplified — actual stats collection needs TrafficManager API)
            cycle_count++;
        }
        
        // Return average latency
        return traffic_manager->GetAverageLatency();
    }
    
    /// === COMPLETE RL STEP ===
    // Run P cycles, return state + reward + done
    py::tuple step(py::array_t<int> action_table) {
        // 1. Apply routing table from AI
        set_routing_table(action_table);
        
        // 2. Run P cycles
        float latency = run_p_cycles(update_period);
        
        // 3. Get new state
        auto new_state = get_network_state();
        
        // 4. Compute reward (negative latency = minimize)
        float reward = -latency;
        
        // 5. Check done
        bool done = (cycle_count >= 100000);
        
        return py::make_tuple(new_state, reward, done);
    }
    
    /// === RESET ===
    py::array_t<float> reset() {
        // Re-initialize simulation
        cycle_count = 0;
        // (full re-init logic needed)
        return get_network_state();
    }
    
    /// === GET STATS ===
    py::dict get_stats() {
        py::dict stats;
        stats["cycle"] = cycle_count;
        stats["avg_latency"] = traffic_manager->GetAverageLatency();
        stats["throughput"] = traffic_manager->GetAcceptedRate();
        return stats;
    }
};


// ============================================================
// PyBind11 Module Definition
// ============================================================
PYBIND11_MODULE(booksim, m) {
    m.doc() = "BookSim2 NoC Simulator — Python Wrapper";
    
    py::class_<BookSimEnv>(m, "NoCEnv")
        .def(py::init<int, std::string, float, int>(),
             py::arg("k") = 4,
             py::arg("traffic") = "hotspot",
             py::arg("inj_rate") = 0.1f,
             py::arg("period") = 5000)
        .def("reset", &BookSimEnv::reset)
        .def("step", &BookSimEnv::step)
        .def("get_state", &BookSimEnv::get_network_state)
        .def("set_routing", &BookSimEnv::set_routing_table)
        .def("run_cycles", &BookSimEnv::run_p_cycles)
        .def("get_stats", &BookSimEnv::get_stats);
}

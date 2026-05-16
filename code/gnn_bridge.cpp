// ============================================================
// GNNocRoute-DRL: BookSim2 ↔ Python Socket Integration
// Phase 1 — Closed-Loop: BookSim2 → Python → BookSim2
// 
// Cách hoạt động:
// 1. BookSim2 gửi state qua socket TCP (JSON)
// 2. Python agent (GNN+PPO) nhận, tính routing policy
// 3. Python gửi routing table về BookSim2
// 4. BookSim2 update và tiếp tục simulation
// ============================================================

#include "booksim.hpp"
#include "routefunc.hpp"
#include "booksim_config.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <string>

// Global socket for Python communication
static int gnn_socket_fd = -1;
static struct sockaddr_in gnn_server_addr;
static bool gnn_connected = false;

// Buffer for routing table: [0]=XY, [1]=YX for each (src,dst)
// Max 256 nodes = 65536 entries
#define GNN_MAX_NODES 256
static int gnn_routing_table[GNN_MAX_NODES][GNN_MAX_NODES];
static int gnn_num_nodes = 0;

// Initialize socket connection to Python server
void gnn_init_socket(const char* ip, int port) {
    gnn_socket_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (gnn_socket_fd < 0) {
        printf("[GNN] Socket creation failed\n");
        return;
    }
    
    gnn_server_addr.sin_family = AF_INET;
    gnn_server_addr.sin_port = htons(port);
    inet_pton(AF_INET, ip, &gnn_server_addr.sin_addr);
    
    // Non-blocking connect (don't block simulation if Python not ready)
    struct timeval tv;
    tv.tv_sec = 0;
    tv.tv_usec = 100000;  // 100ms timeout
    
    int res = connect(gnn_socket_fd, (struct sockaddr*)&gnn_server_addr, sizeof(gnn_server_addr));
    if (res < 0) {
        printf("[GNN] WARNING: Python agent not available (socket connect failed)\n");
        printf("[GNN] Falling back to XY routing\n");
        gnn_connected = false;
        close(gnn_socket_fd);
        gnn_socket_fd = -1;
    } else {
        printf("[GNN] Connected to Python agent at %s:%d\n", ip, port);
        gnn_connected = true;
    }
}

// Send network state to Python, receive routing table
// State: congestion levels for each node
bool gnn_query(float* congestion, int num_nodes) {
    if (!gnn_connected) return false;
    
    // Format: "STATE:<num_nodes>:<c0>,<c1>,...,<cN>"
    char buffer[4096];
    int pos = sprintf(buffer, "STATE:%d:", num_nodes);
    for (int i = 0; i < num_nodes && pos < 4000; i++) {
        pos += sprintf(buffer + pos, "%.4f,", congestion[i]);
    }
    buffer[pos-1] = '\n';  // replace last comma with newline
    
    // Send
    int sent = send(gnn_socket_fd, buffer, strlen(buffer), 0);
    if (sent < 0) {
        printf("[GNN] Send failed\n");
        return false;
    }
    
    // Receive routing table: "TABLE:<N>:<row0>,<row1>,..."
    memset(buffer, 0, sizeof(buffer));
    int received = recv(gnn_socket_fd, buffer, sizeof(buffer)-1, 0);
    if (received <= 0) {
        printf("[GNN] Receive failed\n");
        return false;
    }
    
    // Parse
    buffer[received] = '\0';
    if (buffer[0] != 'T') return false;
    
    char* ptr = buffer + 6;  // skip "TABLE:"
    int n = atoi(ptr);
    if (n != num_nodes) return false;
    
    ptr = strchr(ptr, ':') + 1;
    
    gnn_num_nodes = n;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            // Read next number
            while (*ptr && (*ptr < '0' || *ptr > '9')) ptr++;
            if (!*ptr) break;
            gnn_routing_table[i][j] = atoi(ptr);
            while (*ptr && *ptr != ',' && *ptr != ';') ptr++;
        }
    }
    
    return true;
}

// GNN-PPO routing function (called by BookSim2 for each packet)
void gnn_ppo_closed_loop_mesh(const Router* r, const Flit* f,
                               int in_channel, OutputSet* outputs, bool inject) {
    int vcBegin = 0, vcEnd = gNumVCs - 1;
    (void)in_channel;
    int out_port;
    
    if (inject) {
        out_port = -1;
    } else if (r->GetID() == f->dest) {
        out_port = 2 * gN;
    } else {
        int cur = r->GetID();
        int dest = f->dest;
        
        if (cur < GNN_MAX_NODES && dest < GNN_MAX_NODES) {
            int route = gnn_routing_table[cur][dest];
            if (route == 0)
                out_port = dor_next_mesh(cur, dest, false);
            else
                out_port = dor_next_mesh(cur, dest, true);
        } else {
            out_port = dor_next_mesh(cur, dest, false);
        }
    }
    
    outputs->Clear();
    outputs->AddRange(out_port, vcBegin, vcEnd);
}

// Callback to send state every P cycles
// To be called from traffic manager
void gnn_periodic_update(int cycle, float* congestion, int num_nodes) {
    // Update every 5000 cycles
    if (cycle % 5000 == 0) {
        gnn_query(congestion, num_nodes);
    }
}

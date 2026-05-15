// GNNocRoute-PPO: Precomputed routing policy
#include "booksim.hpp"
#include "routefunc.hpp"
#include "routefunc.hpp"
#include "kncube.hpp"

// Routing table: [src][dst] = 0(XY) or 1(YX)
static const int gnn_route_table[16][16] = {
  {0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
  {0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
  {0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
  {0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
  {0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
  {0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
  {0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1},
  {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1},
  {0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1},
  {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1},
  {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1},
  {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1},
  {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1},
  {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1},
  {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1},
  {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0} 
};

extern int dor_next_mesh(int, int, bool);

void gnn_ppo_route_mesh( const Router *r, const Flit *f, 
         int in_channel, OutputSet *outputs, bool inject )
{
  int vcBegin = 0, vcEnd = gNumVCs-1;
  (void)in_channel;
  int out_port;
  if(inject) {
    out_port = -1;
  } else if(r->GetID() == f->dest) {
    out_port = 2*gN;
  } else {
    int cur = r->GetID();
    int dest = f->dest;
    int route = gnn_route_table[cur][dest];
    
    // Route according to GNN-PPO policy
    if(route == 0) {  // XY
      out_port = dor_next_mesh( cur, dest, false );
    } else {  // YX
      out_port = dor_next_mesh( cur, dest, true );
    }
  }
  outputs->Clear();
  outputs->AddRange( out_port , vcBegin, vcEnd );
}

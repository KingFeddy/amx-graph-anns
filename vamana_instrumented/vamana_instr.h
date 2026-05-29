#pragma once
#include <vector>

struct VamanaHopEntry {
    int query_id;
    int hop;
    int node_id;
};

extern std::vector<VamanaHopEntry> vamana_hop_log;
extern bool vamana_instr_enabled;
extern thread_local int vamana_current_qid;

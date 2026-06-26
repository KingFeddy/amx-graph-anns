#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>

#include <faiss/IndexHNSW.h>
#include <faiss/impl/HNSW.h>
#include <faiss/index_io.h>

std::vector<std::vector<float>> load_fvecs(const std::string& path, int max_n = -1) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::cerr << "Cannot open " << path << "\n";
        exit(1);
    }
    std::vector<std::vector<float>> data;
    while (!f.eof()) {
        int dim;
        f.read((char*)&dim, 4);
        if (f.eof()) break;
        std::vector<float> v(dim);
        f.read((char*)v.data(), dim * 4);
        data.push_back(v);
        if (max_n > 0 && (int)data.size() >= max_n) break;
    }
    return data;
}

int main(int argc, char* argv[]) {
    std::string base_path = argc > 1 ? argv[1] : "data/sift/sift_base.fvecs";
    std::string query_path = argc > 2 ? argv[2] : "data/sift/sift_query.fvecs";
    std::string index_path = argc > 3 ? argv[3] : "faiss_hnsw_m16.bin";
    std::string out_path = argc > 4 ? argv[4] : "faiss_hop_log.csv";

    const int N_QUERIES = 1000;
    const int EF_SEARCH = 100;
    const int M = 16;
    const int EF_CONST = 200;
    const int K = 10;

    std::cout << "[1/5] Loading base vectors...\n";
    auto base = load_fvecs(base_path);
    int N = (int)base.size();
    std::cout << "      Loaded " << N << " vectors\n";

    std::cout << "[2/5] Loading " << N_QUERIES << " queries...\n";
    auto queries = load_fvecs(query_path, N_QUERIES);
    const int DIM = queries.empty() ? 128 : (int)queries[0].size();
    std::cout << "      DIM=" << DIM << "\n";

    std::vector<float> base_flat(N * DIM);
    for (int i = 0; i < N; i++) {
        std::copy(base[i].begin(), base[i].end(), base_flat.data() + i * DIM);
    }

    std::vector<float> query_flat(N_QUERIES * DIM);
    for (int i = 0; i < N_QUERIES; i++) {
        std::copy(queries[i].begin(), queries[i].end(), query_flat.data() + i * DIM);
    }

    faiss::IndexHNSWFlat* index;
    std::ifstream idx_check(index_path);
    if (idx_check.good()) {
        idx_check.close();
        std::cout << "[3/5] Loading saved index " << index_path << "...\n";
        index = dynamic_cast<faiss::IndexHNSWFlat*>(faiss::read_index(index_path.c_str()));
    } else {
        std::cout << "[3/5] Building index (M=" << M
                  << ", ef_construction=" << EF_CONST << ")...\n";
        index = new faiss::IndexHNSWFlat(DIM, M);
        index->hnsw.efConstruction = EF_CONST;
        auto t0 = std::chrono::steady_clock::now();
        index->add(N, base_flat.data());
        auto t1 = std::chrono::steady_clock::now();
        std::cout << "      Built in "
                  << std::chrono::duration<double>(t1 - t0).count()
                  << "s. Saving...\n";
        faiss::write_index(index, index_path.c_str());
    }

    index->hnsw.efSearch = EF_SEARCH;

    std::vector<faiss::idx_t> labels(K);
    std::vector<float> dists(K);

    std::cout << "[4/5] Running " << N_QUERIES << " instrumented queries...\n";
    faiss_instr::hnsw_hop_log.reserve(N_QUERIES * 300);
    faiss_instr::hnsw_instr_enabled = true;

    for (int q = 0; q < N_QUERIES; q++) {
        faiss_instr::hnsw_current_qid = q;
        index->search(1, query_flat.data() + q * DIM, K, dists.data(), labels.data());
        if (q % 200 == 0) {
            std::cout << "      Query " << q << "/" << N_QUERIES << "\n";
        }
    }

    faiss_instr::hnsw_instr_enabled = false;
    std::cout << "      Logged: " << faiss_instr::hnsw_hop_log.size() << " entries\n";

    std::cout << "[5/5] Writing " << out_path << "...\n";
    std::ofstream out(out_path);
    out << "query_id,hop,neighbor_id\n";
    for (auto& e : faiss_instr::hnsw_hop_log) {
        out << e.query_id << "," << e.hop << "," << e.node_id << "\n";
    }

    std::cout << "Done.\n";
    delete index;
    return 0;
}

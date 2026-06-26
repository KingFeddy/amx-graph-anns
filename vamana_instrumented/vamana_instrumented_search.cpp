#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>

#include "index.h"
#include "parameters.h"
#include "vamana_instr.h"

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
    std::string index_path = argc > 3 ? argv[3] : "vamana_sift";
    std::string out_path = argc > 4 ? argv[4] : "vamana_hop_log.csv";

    const int N_QUERIES = argc > 5 ? std::stoi(argv[5]) : 1000;
    const int num_base_pts = argc > 6 ? std::stoi(argv[6]) : -1;
    bool index_prebuilt = std::ifstream(index_path).good();
    const uint32_t K = 10;
    const uint32_t L_SEARCH = 100;
    const uint32_t R = 32;
    const uint32_t L_BUILD = 125;
    const float ALPHA = 1.2f;

    size_t N = 0;
    std::vector<std::vector<float>> base;
    if (!index_prebuilt || num_base_pts < 0) {
        std::cout << "[1/5] Loading base vectors...\n";
        base = load_fvecs(base_path);
        N = base.size();
        std::cout << "      Loaded " << N << " vectors\n";
    } else {
        N = num_base_pts;
        std::cout << "[1/5] Pre-built index found, skipping base load (N=" << N << ")\n";
    }

    std::cout << "[2/5] Loading " << N_QUERIES << " queries...\n";
    auto queries = load_fvecs(query_path, N_QUERIES);
    const uint32_t DIM = queries.empty() ? 128 : (uint32_t)queries[0].size();
    std::cout << "      DIM=" << DIM << "\n";

    std::vector<float> base_flat;
    if (!index_prebuilt || num_base_pts < 0) {
        base_flat.resize(N * DIM);
        for (size_t i = 0; i < N; i++) {
            std::copy(base[i].begin(), base[i].end(), base_flat.data() + i * DIM);
        }
    }

    diskann::IndexWriteParameters params = diskann::IndexWriteParametersBuilder(L_BUILD, R)
        .with_alpha(ALPHA)
        .with_num_threads(1)
        .build();

    auto search_params = std::make_shared<diskann::IndexSearchParams>(L_SEARCH, 1);

    diskann::Index<float> index(diskann::Metric::L2, DIM, N, std::make_shared<diskann::IndexWriteParameters>(params), search_params);

    std::string index_file = index_path;
    if (index_prebuilt) {
        std::cout << "[3/5] Loading saved Vamana index...\n";
        index.load(index_path.c_str(), 1, L_SEARCH);
    } else {
        std::cout << "[3/5] Building Vamana index (R=" << R
                  << ", L=" << L_BUILD << ", alpha=" << ALPHA << ")...\n";
        auto t0 = std::chrono::steady_clock::now();
        index.build(base_flat.data(), N, {});
        auto t1 = std::chrono::steady_clock::now();
        std::cout << "      Built in "
                  << std::chrono::duration<double>(t1 - t0).count()
                  << "s. Saving...\n";
        index.save(index_path.c_str());
    }

    std::vector<uint32_t> labels(K);
    std::vector<float> dists(K);

    std::cout << "[4/5] Running " << N_QUERIES
              << " instrumented queries (L=" << L_SEARCH << ")...\n";
    vamana_hop_log.reserve(N_QUERIES * 300);
    vamana_instr_enabled = true;

    for (int q = 0; q < N_QUERIES; q++) {
        vamana_current_qid = q;
        index.search(queries[q].data(), K, L_SEARCH, labels.data(), dists.data());
        if (q % 200 == 0) {
            std::cout << "      Query " << q << "/" << N_QUERIES << "\n";
        }
    }

    vamana_instr_enabled = false;
    std::cout << "      Logged: " << vamana_hop_log.size() << " entries\n";

    std::cout << "[5/5] Writing " << out_path << "...\n";
    std::ofstream out(out_path);
    out << "query_id,hop,neighbor_id\n";
    for (auto& e : vamana_hop_log) {
        out << e.query_id << "," << e.hop << "," << e.node_id << "\n";
    }

    std::cout << "Done.\n";
    return 0;
}

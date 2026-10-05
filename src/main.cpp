#include "Config.hpp"
#include "Parallel.hpp"
#include "Simulation.hpp"

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#if defined(KOKKOS_ENABLE_CUDA)
#include <cuda_runtime.h>
#elif defined(KOKKOS_ENABLE_HIP)
#include <hip/hip_runtime.h>
#endif

// number of GPUs visible to this process (0 for host backends)
static int visible_devices() {
    int n = 0;
#if defined(KOKKOS_ENABLE_CUDA)
    if (cudaGetDeviceCount(&n) != cudaSuccess) n = 0;
#elif defined(KOKKOS_ENABLE_HIP)
    if (hipGetDeviceCount(&n) != hipSuccess) n = 0;
#endif
    return n;
}

int main(int argc, char* argv[]) {
    // MPI first (Kokkos must be initialised after MPI and finalised before it)
    qsrz::Comm::init(&argc, &argv);
    qsrz::Comm& comm = qsrz::Comm::world();
    int rc = 0;
    {
        // one GPU per rank: device = (rank within the node) mod (#GPUs), unless the user
        // chose one (--kokkos-device-id=..., KOKKOS_DEVICE_ID, or CUDA_VISIBLE_DEVICES per rank)
        std::vector<char*> args(argv, argv + argc);
        std::string devarg;
        bool user_dev = std::getenv("KOKKOS_DEVICE_ID") != nullptr;
        for (int i = 1; i < argc; ++i)
            if (std::strncmp(argv[i], "--kokkos-device-id", 18) == 0 || std::strncmp(argv[i], "--kokkos-map-device-id-by", 25) == 0)
                user_dev = true;
        const int ndev = visible_devices();
        if (!user_dev && ndev > 0) {
            devarg = "--kokkos-device-id=" + std::to_string(comm.local_rank() % ndev);
            args.push_back(devarg.data());
        }
        args.push_back(nullptr);
        int kargc = static_cast<int>(args.size()) - 1;
        Kokkos::ScopeGuard kokkos(kargc, args.data());

        if (argc < 2) {
            if (comm.root()) std::cerr << "usage: [mpirun -np P] qsrz <input file> [key=value ...] [--kokkos-... options]\n";
            rc = 1;
        } else {
            try {
                qsrz::Config cfg(argv[1]);
                // command line overrides:  key=value
                for (int i = 2; i < argc; ++i) {
                    std::string a = argv[i];
                    auto eq = a.find('=');
                    if (a.rfind("--", 0) == 0 || eq == std::string::npos) continue;
                    cfg.set(a.substr(0, eq), a.substr(eq + 1));
                }
                if (comm.root() && comm.size() > 1)
                    std::cout << "MPI: " << comm.size() << " ranks (decomposition along xi)"
                              << (ndev > 0 ? ", " + std::to_string(ndev) + " GPU(s) per node" : std::string()) << "\n";
                {
                    qsrz::Simulation sim(cfg);
                    sim.run();
                }
                if (comm.root())
                    for (const auto& k : cfg.unused_keys()) std::cout << "WARNING: input key '" << k << "' was not used\n";
            } catch (const std::exception& e) {
                std::cerr << "ERROR (rank " << comm.rank() << "): " << e.what() << "\n";
                rc = 2;
            }
        }
    }
    if (rc == 2 && comm.size() > 1) qsrz::Comm::abort(2);   // other ranks may wait in a receive
    qsrz::Comm::finalize();
    return rc;
}

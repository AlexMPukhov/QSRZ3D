// Thin communication layer.  With QUARZ_USE_MPI it wraps MPI; without it, it is
// a serial stub (rank 0 of 1), so the rest of the code has a single code path.
//
// Used for the pipelined decomposition along xi: messages are plain buffers of
// doubles sent from rank r to rank r+1 once per time step, and files are written
// at byte offsets by each rank independently (no collective I/O, which would
// synchronise the pipeline).
#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace quarz {

class Comm {
public:
    static void init(int* argc, char*** argv);   // MPI_Init (no-op without MPI)
    static void finalize();
    [[noreturn]] static void abort(int code);    // MPI_Abort / std::exit
    static Comm& world();

    int rank() const { return rank_; }
    int size() const { return size_; }
    bool root() const { return rank_ == 0; }
    int local_rank() const { return local_rank_; }   // rank within the node (GPU selection)

    // non-blocking send of a buffer to dest; the previous send is completed first
    // (the buffer is kept alive inside Comm until the next call or wait_send)
    void isend(int dest, int tag, std::vector<double>&& buf);
    void wait_send();
    // blocking receive of a message of unknown length
    std::vector<double> recv(int src, int tag);
    void barrier();
    double allreduce_max(double v);

    // write `bytes` at byte `offset` of file fn (created if missing); if set_size >= 0
    // the file is first set to that size (truncate/extend)
    static void write_at(const std::string& fn, long long offset, const void* data, size_t bytes,
                         long long set_size = -1);

private:
    int rank_ = 0, size_ = 1, local_rank_ = 0;
    std::vector<double> sendbuf_;
    void* req_ = nullptr;   // MPI_Request (opaque here)
    bool pending_ = false;
};

} // namespace quarz

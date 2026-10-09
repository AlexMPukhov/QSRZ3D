#include "Parallel.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>

#ifdef QUARZ_USE_MPI
#include <mpi.h>
#endif

namespace quarz {

Comm& Comm::world() {
    static Comm c;
    return c;
}

void Comm::init(int* argc, char*** argv) {
    Comm& c = world();
#ifdef QUARZ_USE_MPI
    int provided;
    MPI_Init_thread(argc, argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &c.rank_);
    MPI_Comm_size(MPI_COMM_WORLD, &c.size_);
    MPI_Comm node;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, c.rank_, MPI_INFO_NULL, &node);
    MPI_Comm_rank(node, &c.local_rank_);
    MPI_Comm_free(&node);
    c.req_ = new MPI_Request(MPI_REQUEST_NULL);
#else
    (void)argc; (void)argv;
#endif
}

void Comm::finalize() {
#ifdef QUARZ_USE_MPI
    Comm& c = world();
    c.wait_send();
    delete static_cast<MPI_Request*>(c.req_);
    c.req_ = nullptr;
    MPI_Finalize();
#endif
}

void Comm::abort(int code) {
#ifdef QUARZ_USE_MPI
    MPI_Abort(MPI_COMM_WORLD, code);
#endif
    std::exit(code);
}

void Comm::wait_send() {
#ifdef QUARZ_USE_MPI
    if (pending_) {
        MPI_Wait(static_cast<MPI_Request*>(req_), MPI_STATUS_IGNORE);
        pending_ = false;
    }
#endif
    test_small(true);
}

void Comm::test_small(bool wait) {
#ifdef QUARZ_USE_MPI
    std::vector<Small> keep;
    for (auto& s : small_) {
        MPI_Request* r = static_cast<MPI_Request*>(s.req);
        int done = 0;
        if (wait) { MPI_Wait(r, MPI_STATUS_IGNORE); done = 1; }
        else MPI_Test(r, &done, MPI_STATUS_IGNORE);
        if (done) delete r;
        else keep.push_back(std::move(s));
    }
    small_ = std::move(keep);
#else
    (void)wait;
#endif
}

void Comm::isend_small(int dest, int tag, std::vector<double>&& buf) {
#ifdef QUARZ_USE_MPI
    test_small(false);
    Small s;
    s.buf = std::move(buf);
    small_.push_back(std::move(s));
    Small& b = small_.back();
    MPI_Request* r = new MPI_Request;
    b.req = r;
    MPI_Isend(b.buf.data(), static_cast<int>(b.buf.size()), MPI_DOUBLE, dest, tag, MPI_COMM_WORLD, r);
#else
    (void)dest; (void)tag; (void)buf;
    throw std::runtime_error("Comm::isend_small without MPI");
#endif
}

void Comm::isend(int dest, int tag, std::vector<double>&& buf) {
#ifdef QUARZ_USE_MPI
    wait_send();
    sendbuf_ = std::move(buf);
    MPI_Isend(sendbuf_.data(), static_cast<int>(sendbuf_.size()), MPI_DOUBLE, dest, tag, MPI_COMM_WORLD,
              static_cast<MPI_Request*>(req_));
    pending_ = true;
#else
    (void)dest; (void)tag; (void)buf;
    throw std::runtime_error("Comm::isend without MPI");
#endif
}

std::vector<double> Comm::recv(int src, int tag) {
#ifdef QUARZ_USE_MPI
    MPI_Status st;
    MPI_Probe(src, tag, MPI_COMM_WORLD, &st);
    int n = 0;
    MPI_Get_count(&st, MPI_DOUBLE, &n);
    std::vector<double> b(static_cast<size_t>(n));
    MPI_Recv(b.data(), n, MPI_DOUBLE, src, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    return b;
#else
    (void)src; (void)tag;
    throw std::runtime_error("Comm::recv without MPI");
#endif
}

void Comm::barrier() {
#ifdef QUARZ_USE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif
}

double Comm::allreduce_min(double v) {
#ifdef QUARZ_USE_MPI
    double out;
    MPI_Allreduce(&v, &out, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    return out;
#else
    return v;
#endif
}

double Comm::allreduce_max(double v) {
#ifdef QUARZ_USE_MPI
    double out;
    MPI_Allreduce(&v, &out, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    return out;
#else
    return v;
#endif
}

void Comm::write_at(const std::string& fn, long long offset, const void* data, size_t bytes, long long set_size) {
#ifdef QUARZ_USE_MPI
    MPI_File fh;
    std::string name = fn;
    if (MPI_File_open(MPI_COMM_SELF, name.data(), MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh) != MPI_SUCCESS)
        throw std::runtime_error("cannot open " + fn + " for writing");
    if (set_size >= 0) MPI_File_set_size(fh, static_cast<MPI_Offset>(set_size));
    const char* p = static_cast<const char*>(data);
    size_t done = 0;
    while (done < bytes) {   // chunks below 2 GB (int count)
        const size_t n = std::min(bytes - done, static_cast<size_t>(1) << 30);
        MPI_File_write_at(fh, static_cast<MPI_Offset>(offset + static_cast<long long>(done)), p + done,
                          static_cast<int>(n), MPI_BYTE, MPI_STATUS_IGNORE);
        done += n;
    }
    MPI_File_close(&fh);
#else
    if (!std::filesystem::exists(fn)) { std::ofstream create(fn, std::ios::binary); }
    if (set_size >= 0) std::filesystem::resize_file(fn, static_cast<std::uintmax_t>(set_size));
    std::fstream f(fn, std::ios::in | std::ios::out | std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + fn + " for writing");
    f.seekp(offset);
    f.write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes));
#endif
}

} // namespace quarz

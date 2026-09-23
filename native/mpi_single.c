/* Single-rank MPI shim for AMICA (epic #165, phase 1). See mpi_single.f90 for
 * the rationale. These implement the ~11 MPI subroutines amica15.f90 calls, for
 * exactly one rank, so no real MPI runtime is linked and the binary is portable.
 *
 * ABI: gfortran calls `call MPI_FOO(...)` as `mpi_foo_(...)` with every argument
 * by reference (pointer). The Fortran source passes generically-typed buffers
 * (integer/double/logical/character) to BCAST/REDUCE/GATHER; because these are
 * external procedures with no explicit interface, gfortran appends any hidden
 * CHARACTER length arguments AFTER the last declared argument (ierr). We read
 * only the fixed leading positions and ignore anything trailing, so a character
 * actual argument is handled the same as any other.
 *
 * The MPI datatype constants (mpi_single.f90) equal the element size in bytes,
 * so the single-rank reduce/gather copy is memcpy(recv, send, count*datatype).
 */
#include <string.h>

/* MPI_INIT / MPI_FINALIZE / MPI_BARRIER: nothing to do for one rank. */
void mpi_init_(int *ierr) { *ierr = 0; }
void mpi_finalize_(int *ierr) { *ierr = 0; }
void mpi_barrier_(const int *comm, int *ierr) {
    (void)comm;
    *ierr = 0;
}

/* This process is always rank 0 of a size-1 world. */
void mpi_comm_rank_(const int *comm, int *rank, int *ierr) {
    (void)comm;
    *rank = 0;
    *ierr = 0;
}
void mpi_comm_size_(const int *comm, int *size, int *ierr) {
    (void)comm;
    *size = 1;
    *ierr = 0;
}

/* Splitting a size-1 communicator yields an equivalent size-1 communicator; the
 * handle value is opaque to the caller, so return it unchanged. */
void mpi_comm_split_(const int *comm, const int *color, const int *key,
                     int *newcomm, int *ierr) {
    (void)color;
    (void)key;
    *newcomm = *comm;
    *ierr = 0;
}

/* Single rank always owns the data; no name buffer to fill beyond a label. The
 * hidden Fortran length of `name` is passed last; honor it to avoid overflow. */
void mpi_get_processor_name_(char *name, int *resultlen, int *ierr,
                             long name_len) {
    static const char label[] = "localhost";
    long n = (long)sizeof(label) - 1;  /* exclude the terminating NUL */
    if (n > name_len) n = name_len;
    memcpy(name, label, (size_t)n);
    /* Fortran CHARACTER is blank-padded, not NUL-terminated. */
    if (n < name_len) memset(name + n, ' ', (size_t)(name_len - n));
    *resultlen = (int)n;
    *ierr = 0;
}

/* Broadcast from root to itself is a no-op (the buffer already holds the value).
 * Buffer type is irrelevant; we never touch it. */
void mpi_bcast_(void *buf, const int *count, const int *datatype,
                const int *root, const int *comm, int *ierr) {
    (void)buf;
    (void)count;
    (void)datatype;
    (void)root;
    (void)comm;
    *ierr = 0;
}

/* Reduce/Allreduce over one rank: the result equals the local contribution, so
 * copy sendbuf -> recvbuf. datatype == element size in bytes. */
void mpi_reduce_(const void *sendbuf, void *recvbuf, const int *count,
                 const int *datatype, const int *op, const int *root,
                 const int *comm, int *ierr) {
    (void)op;
    (void)root;
    (void)comm;
    memcpy(recvbuf, sendbuf, (size_t)(*count) * (size_t)(*datatype));
    *ierr = 0;
}
void mpi_allreduce_(const void *sendbuf, void *recvbuf, const int *count,
                    const int *datatype, const int *op, const int *comm,
                    int *ierr) {
    (void)op;
    (void)comm;
    memcpy(recvbuf, sendbuf, (size_t)(*count) * (size_t)(*datatype));
    *ierr = 0;
}

/* Gather to one rank: recvbuf gets this rank's send block. sendtype == bytes. */
void mpi_gather_(const void *sendbuf, const int *sendcount, const int *sendtype,
                 void *recvbuf, const int *recvcount, const int *recvtype,
                 const int *root, const int *comm, int *ierr) {
    (void)recvcount;
    (void)recvtype;
    (void)root;
    (void)comm;
    memcpy(recvbuf, sendbuf, (size_t)(*sendcount) * (size_t)(*sendtype));
    *ierr = 0;
}

#!/usr/bin/env python3
"""Patch the build copy of amica15 for a portable, seedable gfortran build.

Applies sccn/amica PR #54 (supersedes #53): (1) size the ``random_seed(PUT=...)``
array via ``random_seed(SIZE=...)`` so it builds under gfortran, not only ifort;
(2) wire a ``seed`` ``input.param`` option so a run's random initialization is
reproducible when set, and clock-random otherwise. Idempotent and verified -- if
an anchor is absent (source drift) it fails loudly rather than building a subtly
wrong binary.

The tracked ``amica15.f90`` now carries PR #54 already (issue #228): pamica
maintains an improved copy rather than mirroring sccn/amica master, because the
unpatched reference cannot be seeded and so cannot anchor a parity gate. These
patches are therefore normally no-ops, and are kept so the script still works on
an unpatched upstream source.
"""

import sys

# (1) header: SIZE-based seed array + the `seed` param state.
HDR_ANCHOR = "integer :: ii, jj, kk, c0, c1, c2"
HDR_ADD = (
    HDR_ANCHOR
    + "\nINTEGER :: nseed, input_seed = 0"
    + "\nLOGICAL :: use_seed = .false."
    + "\nINTEGER, ALLOCATABLE :: seedvec(:)"
)

# (2) portable + reproducible seeding.
OLD_SEED = "call random_seed(PUT = c1 * (myrank+1) * (seed+myrank+1))"
NEW_SEED = """\
! Portable + reproducible RNG seeding (sccn/amica PR #54, supersedes #53):
! random_seed(PUT=...) needs an array of the compiler SIZE (query with SIZE=);
! the original size-2 array only compiled under ifort. With `seed` set in the
! param file, seed deterministically (reproducible); otherwise clock-seed.
call random_seed(size = nseed)
allocate(seedvec(nseed))
if (use_seed) then
   do jj = 1, nseed
      seedvec(jj) = input_seed + 1009*myrank + 37*(jj-1)
   end do
else
   do jj = 1, nseed
      seedvec(jj) = c1 + 1009*(myrank+1)*jj + 37*(jj-1)
   end do
end if
call random_seed(PUT = seedvec)
deallocate(seedvec)"""

# (3) broadcast the new param state to all ranks.
OLD_BCAST = "call MPI_BCAST(rholratefact,1,MPI_DOUBLE_PRECISION,0,seg_comm,ierr)"
NEW_BCAST = (
    OLD_BCAST
    + "\ncall MPI_BCAST(input_seed,1,MPI_INTEGER,0,seg_comm,ierr)"
    + "\ncall MPI_BCAST(use_seed,1,MPI_LOGICAL,0,seg_comm,ierr)"
)

# (4) parse the `seed` option from input.param (appended after the `indir` case).
OLD_PARSE = "     case('indir')\n        read(tmparg,'(a)') indirparam"
NEW_PARSE = (
    OLD_PARSE
    + "\n     case('seed')"
    + "\n        read(tmparg,'(i12)') input_seed"
    + "\n        use_seed = .true."
    + "\n        print *, 'seed = ', input_seed; call flush(6)"
)


def patch(path, old, new, marker, label):
    """Replace `old` with `new` in `path`. `marker` is a token unique to the
    patched form; if already present, this is a no-op (idempotent)."""
    with open(path) as f:
        text = f.read()
    if marker in text:
        return  # already patched
    if old not in text:
        sys.exit(f"ERROR: {label} anchor not found in {path}; source may have drifted")
    with open(path, "w") as f:
        f.write(text.replace(old, new, 1))


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    pin_seed = "--pin-seed" in sys.argv[1:]
    amica15, header = args[0], args[1]

    patch(header, HDR_ANCHOR, HDR_ADD, "INTEGER :: nseed", "header seedvec decl")
    patch(amica15, OLD_SEED, NEW_SEED, "call random_seed(size = nseed)", "random_seed")
    patch(amica15, OLD_BCAST, NEW_BCAST, "MPI_BCAST(use_seed", "seed BCAST")
    patch(amica15, OLD_PARSE, NEW_PARSE, "case('seed')", "seed param parser")

    if pin_seed:
        # Determinism for validate_shim.sh: drop the clock term (c1) from the
        # default (unseeded) branch so the shim and mpif90 builds seed identically
        # and can be compared bit-for-bit. Applied to the file after patching, not
        # to NEW_SEED, so it still takes effect when the source already carries
        # PR #54 and the patch above was a no-op.
        text = open(amica15).read()
        if "seedvec(jj) = c1 +" not in text:
            sys.exit(
                f"ERROR: --pin-seed anchor 'seedvec(jj) = c1 +' not found in {amica15}"
            )
        with open(amica15, "w") as f:
            f.write(text.replace("seedvec(jj) = c1 +", "seedvec(jj) = 987654321 +", 1))
    print(
        "patched: portable+reproducible random_seed, seed param, BCAST"
        f"{' (pinned)' if pin_seed else ''}"
    )


if __name__ == "__main__":
    main()

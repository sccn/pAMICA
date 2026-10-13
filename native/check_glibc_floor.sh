#!/usr/bin/env bash
# Fail when a Linux binary needs a newer glibc than a stated floor, or shared
# libraries outside an allowed set.
#
# A binary built on a recent distribution silently requires that distribution's glibc
# (issue #370): the loader on an older host refuses it. The release job builds the
# Linux targets in a glibc 2.28 container and runs this check, so the floor cannot
# creep up again when a toolchain, base image, or runner changes.
#
# Usage:
#   bash native/check_glibc_floor.sh <binary> <max-glibc-version> [<allowed-needed-regex>]
#   bash native/check_glibc_floor.sh dist/amica15-linux-x64 2.28
#   bash native/check_glibc_floor.sh dist/amica15-linux-x64 2.28 '^(libc|libm|libdl|libpthread|libmvec)\.so'
#
# The optional regex is matched against each shared library the binary lists as
# needed (its DT_NEEDED entries); any library that does not match fails the check.
# This is how the release job keeps a runtime such as libgomp.so.1 from creeping
# back into a binary that is meant to carry it statically.
#
# Exit status: 0 when every check passes, 1 when one fails, 2 on bad usage or when the
# binary cannot be inspected. A binary that cannot be read is never reported as passing.
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 <binary> <max-glibc-version> [<allowed-needed-regex>]" >&2
  exit 2
fi
bin="$1"
max="$2"
allowed="${3:-}"
if [[ ! -f "$bin" ]]; then
  echo "ERROR: not a file: $bin" >&2
  exit 2
fi
if [[ "$(head -c 4 "$bin")" != $'\x7f'ELF ]]; then
  echo "ERROR: not an ELF binary: $bin" >&2
  exit 2
fi
if ! [[ "$max" =~ ^[0-9]+(\.[0-9]+)+$ ]]; then
  echo "ERROR: floor must look like 2.28, got '$max'" >&2
  exit 2
fi

if command -v readelf >/dev/null 2>&1; then
  inspector=readelf
elif command -v objdump >/dev/null 2>&1; then
  inspector=objdump
else
  echo "ERROR: need readelf or objdump (binutils)" >&2
  exit 2
fi

# readelf exits 0 on a truncated or damaged file and reports "Error:" on stderr instead,
# while objdump exits 1; either way a file that cannot be read must not pass.
errs="$(mktemp)"
trap 'rm -f "$errs"' EXIT
inspector_failed() {  # $1 = exit status of the inspector; stderr of the call is in $errs
  [[ "$1" != 0 ]] || grep -qiE '(error|warning):|not recognized' "$errs"
}

# Program headers, to tell a static binary from a dynamic one whose symbol table
# could not be read.
rc=0
if [[ "$inspector" == readelf ]]; then
  phdrs="$(readelf -lW "$bin" 2>"$errs")" || rc=$?
else
  phdrs="$(objdump -p "$bin" 2>"$errs")" || rc=$?
fi
if inspector_failed "$rc"; then
  echo "ERROR: cannot read the program headers of $bin: $(sed -n '1p' "$errs")" >&2
  exit 2
fi
dynamic=0
if grep -q 'INTERP' <<<"$phdrs"; then dynamic=1; fi

# Dynamic symbol table, one symbol per line. A static binary has none, and the
# inspectors may complain about that; the complaint only matters for a dynamic one.
rc=0
if [[ "$inspector" == readelf ]]; then
  table="$(readelf --dyn-syms --wide "$bin" 2>"$errs")" || rc=$?
else
  table="$(objdump -T "$bin" 2>"$errs")" || rc=$?
fi
if [[ "$dynamic" == 1 ]] && inspector_failed "$rc"; then
  echo "ERROR: cannot read the dynamic symbols of $bin: $(sed -n '1p' "$errs")" >&2
  exit 2
fi

# Versions only (2.28, not GLIBC_PRIVATE), highest first.
versions="$(grep -oE 'GLIBC_[0-9]+(\.[0-9]+)+' <<<"$table" | sed 's/^GLIBC_//' | sort -u || true)"
# A binary linked with -z pack-relative-relocs asks for the version GLIBC_ABI_DT_RELR,
# which has no number and appears in the version requirements, not among the symbols;
# glibc added support for DT_RELR in 2.36.
relr=0
if [[ "$dynamic" == 1 ]]; then
  if [[ "$inspector" == readelf ]]; then verneed="$(readelf -V "$bin" 2>/dev/null || true)"; else verneed="$phdrs"; fi
  if grep -q 'GLIBC_ABI_DT_RELR' <<<"$verneed"; then
    relr=1
    versions="$(printf '%s\n2.36\n' "$versions" | sed '/^$/d' | sort -u)"
  fi
fi
versions="$(sort -rV <<<"$versions" | sed '/^$/d')"

status=0
if [[ -z "$versions" ]]; then
  if [[ "$dynamic" == 1 ]]; then
    echo "ERROR: $bin is dynamically linked but no versioned glibc symbols could be read; cannot vouch for its floor" >&2
    exit 2
  fi
  echo "glibc floor OK: $bin is static and references no glibc symbols"
else
  highest="$(sed -n '1p' <<<"$versions")"
  if [[ "$(printf '%s\n%s\n' "$highest" "$max" | sort -V | tail -1)" == "$max" ]]; then
    echo "glibc floor OK: $bin needs glibc $highest or newer (floor $max)"
  else
    echo "ERROR: $bin needs glibc $highest, above the floor $max. Symbols above the floor:" >&2
    while read -r v; do
      if [[ "$(printf '%s\n%s\n' "$v" "$max" | sort -V | tail -1)" != "$max" ]]; then
        if [[ "$v" == 2.36 && "$relr" == 1 ]]; then
          echo "  (relative relocations packed as DT_RELR, needs glibc 2.36)" >&2
        fi
        # objdump -T ends each line with the symbol. readelf --dyn-syms prints it as name@version,
        # followed by a "(n)" index and, on arm64, flags such as [VARIANT_PCS].
        # sed -n 1,5p, not head -5: head exits after five lines, and under pipefail the
        # SIGPIPE the upstream stages then take would end the script with status 141.
        grep -E "GLIBC_${v//./\\.}([^0-9.]|\$)" <<<"$table" \
          | awk -v v="$v" '{n = $NF; for (i = 1; i <= NF; i++) if ($i ~ /@/) n = $i; sub(/@.*/, "", n); print "  " n " (GLIBC_" v ")"}' \
          | sort -u | sed -n '1,5p' >&2
      fi
    done <<<"$versions"
    status=1
  fi
fi

if [[ -n "$allowed" ]]; then
  rc=0
  if [[ "$inspector" == readelf ]]; then
    dyn="$(readelf -dW "$bin" 2>"$errs")" || rc=$?
    needed="$(sed -nE 's/.*\(NEEDED\).*\[(.*)\].*/\1/p' <<<"$dyn")"
  else
    needed="$(awk '$1 == "NEEDED" {print $2}' <<<"$phdrs")"
  fi
  if [[ "$dynamic" == 1 ]] && inspector_failed "$rc"; then
    echo "ERROR: cannot read the needed libraries of $bin: $(sed -n '1p' "$errs")" >&2
    exit 2
  fi
  unexpected=""
  while read -r lib; do
    if [[ -z "$lib" ]]; then continue; fi
    if ! grep -qE "$allowed" <<<"$lib"; then unexpected+="  $lib"$'\n'; fi
  done <<<"$needed"
  if [[ -n "$unexpected" ]]; then
    echo "ERROR: $bin needs shared libraries outside the allowed set ($allowed):" >&2
    printf '%s' "$unexpected" >&2
    status=1
  else
    echo "shared libraries OK: $(tr '\n' ' ' <<<"$needed" | sed 's/ *$//')"
  fi
fi
exit "$status"

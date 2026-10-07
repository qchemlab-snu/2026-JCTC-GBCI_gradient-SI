#!/bin/bash
# Render the active-orbital cubes written by render_active_orbitals.py with Jmol.
#
# Run render_active_orbitals.py first: it writes the cubes and the manifest that
# names the views each system needs.  Run from this directory; all paths are
# relative to it.
#
# Jmol on this cluster always needs an X display, even with -n/-o and
# java.awt.headless, so a virtual framebuffer is started here and torn down on
# exit.  Point XVFB_BIN at a different Xvfb if yours lives elsewhere.
set -euo pipefail

FIG_DIR="${FIG_DIR:-reference/active_orbital_figures}"
SPT_TEMPLATE="render_active_orbitals_jmol.spt"
JMOL_MODULE="jmol/16.1.47"
XVFB_BIN="${XVFB_BIN:-$HOME/anaconda3/envs/xvfb/x86_64-conda-linux-gnu/sysroot/usr/bin/Xvfb}"
XVFB_LIBS="${XVFB_LIBS:-$HOME/anaconda3/envs/xvfb/x86_64-conda-linux-gnu/sysroot/usr/lib64:$HOME/anaconda3/envs/xvfb/lib}"
XVFB_DISPLAY="${XVFB_DISPLAY:-:91}"

if [ "$#" -gt 0 ]; then
    SYSTEMS="${*//,/ }"
else
    SYSTEMS="$(cd "$FIG_DIR" && ls -d */ 2>/dev/null | tr -d /)"
fi
if [ -z "$SYSTEMS" ]; then
    echo "error: no systems under $FIG_DIR; run render_active_orbitals.py first" >&2
    exit 1
fi

XVFB_PID=""
cleanup() {
    [ -n "$XVFB_PID" ] && kill "$XVFB_PID" 2>/dev/null || true
}
trap cleanup EXIT

if [ -z "${DISPLAY:-}" ]; then
    if [ ! -x "$XVFB_BIN" ]; then
        echo "error: no DISPLAY and no Xvfb at $XVFB_BIN" >&2
        exit 1
    fi
    LD_LIBRARY_PATH="$XVFB_LIBS" "$XVFB_BIN" "$XVFB_DISPLAY" -screen 0 1400x1100x24 \
        >/dev/null 2>&1 &
    XVFB_PID=$!
    export DISPLAY="$XVFB_DISPLAY"
    sleep 3
fi

# `module` is a shell function, so a non-interactive shell does not inherit it.
# Both profile scripts are needed: modules.sh defines the function but points
# MODULEPATH at ~/modules only, and zz-site-modules.sh appends /appl/modulefiles,
# which is where jmol lives.  Every one of these ends on a `test 0 = 1`, so they
# report failure even on success and must not trip `set -e`.
if ! command -v module >/dev/null; then
    # shellcheck disable=SC1091
    . /etc/profile.d/modules.sh || true
    # shellcheck disable=SC1091
    . /etc/profile.d/zz-site-modules.sh || true
fi
module add "$JMOL_MODULE" || true
if ! command -v jmol >/dev/null; then
    echo "error: jmol not on PATH after loading $JMOL_MODULE" >&2
    exit 1
fi

for system in $SYSTEMS; do
    manifest="$FIG_DIR/$system/${system}_manifest.json"
    if [ ! -f "$manifest" ]; then
        echo "error: $manifest not found; run render_active_orbitals.py --systems $system" >&2
        exit 1
    fi
    # The manifest, not the directory listing, decides which views a system gets:
    # a linear molecule is rendered front-on only.
    views="$(python3 -c "import json,sys;print(' '.join(json.load(open(sys.argv[1]))['views']))" "$manifest")"
    image_dir="$FIG_DIR/$system/orbitals"
    mkdir -p "$image_dir"

    for cube in "$FIG_DIR/$system"/cubes/*.cube; do
        stem="$(basename "$cube" .cube)"
        for view in $views; do
            case "$view" in
                front) rotate="" ;;
                side)  rotate="rotate x 90" ;;
                *) echo "error: unknown view '$view'" >&2; exit 1 ;;
            esac
            png="$image_dir/${stem}_${view}.png"
            spt="$image_dir/.${stem}_${view}.spt"
            sed -e "s|__INFILE__|$cube|g" -e "s|__OUTFILE__|$png|" \
                -e "s|__ROTATE__|$rotate|" "$SPT_TEMPLATE" > "$spt"
            jmol -ionx -s "$spt" >/dev/null 2>&1
            rm -f "$spt"
            if [ -s "$png" ]; then
                echo "wrote $png"
            else
                echo "error: Jmol produced no image for $cube ($view)" >&2
                exit 1
            fi
        done
    done
done

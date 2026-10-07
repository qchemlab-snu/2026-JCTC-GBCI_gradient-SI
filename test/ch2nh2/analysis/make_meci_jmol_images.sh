#!/bin/bash
# Render the CH2NH2 cc-pVTZ S0/S1 MECI geometries with Jmol.
#
# Run ch2nh2_meci_geometry_descriptors.py first: it writes the reoriented *_aligned.xyz
# files that this script renders.  Run from this directory; all paths are relative to it.
#
# Jmol on this cluster always needs an X display, even with -n/-o and java.awt.headless,
# so a virtual framebuffer is started here and torn down on exit.  Point XVFB_BIN at a
# different Xvfb if yours lives elsewhere.
set -euo pipefail

IMAGE_DIR="outputs/ch2nh2_ccpvtz_meci_images"
SPT_TEMPLATE="render_meci_jmol.spt"
JMOL_MODULE="jmol/16.1.47"
XVFB_BIN="${XVFB_BIN:-$HOME/anaconda3/envs/xvfb/x86_64-conda-linux-gnu/sysroot/usr/bin/Xvfb}"
XVFB_LIBS="${XVFB_LIBS:-$HOME/anaconda3/envs/xvfb/x86_64-conda-linux-gnu/sysroot/usr/lib64:$HOME/anaconda3/envs/xvfb/lib}"
XVFB_DISPLAY="${XVFB_DISPLAY:-:91}"

if [ ! -d "$IMAGE_DIR" ]; then
    echo "error: $IMAGE_DIR not found; run ch2nh2_meci_geometry_descriptors.py first" >&2
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
    LD_LIBRARY_PATH="$XVFB_LIBS" "$XVFB_BIN" "$XVFB_DISPLAY" -screen 0 1200x900x24 \
        >/dev/null 2>&1 &
    XVFB_PID=$!
    export DISPLAY="$XVFB_DISPLAY"
    sleep 3
fi

# The module function ends on a `test 0 = 1`, so it reports failure even on success.
module add "$JMOL_MODULE" || true
if ! command -v jmol >/dev/null; then
    echo "error: jmol not on PATH after loading $JMOL_MODULE" >&2
    exit 1
fi

for xyz in "$IMAGE_DIR"/*_aligned.xyz; do
    stem="$(basename "$xyz" _aligned.xyz)"
    png="$IMAGE_DIR/$stem.png"
    spt="$IMAGE_DIR/.$stem.spt"
    sed -e "s|__INFILE__|$xyz|" -e "s|__OUTFILE__|$png|" "$SPT_TEMPLATE" > "$spt"
    jmol -ionx -s "$spt" >/dev/null 2>&1
    rm -f "$spt"
    if [ -s "$png" ]; then
        echo "wrote $png"
    else
        echo "error: Jmol produced no image for $xyz" >&2
        exit 1
    fi
done

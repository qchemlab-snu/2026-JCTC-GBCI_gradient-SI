#!/bin/bash
# Render the SA-CASSCF NEB endpoints as the structure figures used by
# pp_casscf_s1_neb_pes_plot_figure.py.
#
# Run align_neb_endpoints_to_reference.py first: it rotates each endpoint into
# the frame of the published CASCI structure it corresponds to. The cameras
# below were then chosen to reproduce how S0_casci_S0_GM.png and
# S1_twisted_LM.png are composed, so the same structure looks the same across
# the paper:
#
#   start  edge-on along the molecular plane, pyrrole left, pyridine right
#   end    the aligned frame as it stands, which is already that composition
#
# Run from this directory. Jmol on this cluster needs an X display even for
# offscreen writes, so a virtual framebuffer is started here and torn down on
# exit. Mirrors test/make_active_orbital_images.sh.
set -euo pipefail

SPT_TEMPLATE="render_molecule_jmol.spt"
JMOL_MODULE="jmol/16.1.47"
XVFB_BIN="${XVFB_BIN:-$HOME/anaconda3/envs/xvfb/x86_64-conda-linux-gnu/sysroot/usr/bin/Xvfb}"
XVFB_LIBS="${XVFB_LIBS:-$HOME/anaconda3/envs/xvfb/x86_64-conda-linux-gnu/sysroot/usr/lib64:$HOME/anaconda3/envs/xvfb/lib}"
XVFB_DISPLAY="${XVFB_DISPLAY:-:92}"

XVFB_PID=""
cleanup() { [ -n "$XVFB_PID" ] && kill "$XVFB_PID" 2>/dev/null || true; }
trap cleanup EXIT

if [ -z "${DISPLAY:-}" ]; then
    LD_LIBRARY_PATH="$XVFB_LIBS" "$XVFB_BIN" "$XVFB_DISPLAY" -screen 0 1400x1100x24 \
        >/dev/null 2>&1 &
    XVFB_PID=$!
    export DISPLAY="$XVFB_DISPLAY"
    sleep 3
fi

if ! command -v module >/dev/null; then
    . /etc/profile.d/modules.sh || true
    . /etc/profile.d/zz-site-modules.sh || true
fi
module add "$JMOL_MODULE" || true
command -v jmol >/dev/null || { echo "error: jmol not on PATH" >&2; exit 1; }

# $1 xyz  $2 png  $3 "keep"/"drop" the `rotate best` line  $4... extra rotations
render () {
    local infile="$1" png="$2" best="$3"; shift 3
    local spt=".render_$(basename "$png" .png).spt"
    awk -v extra="$*" '{ if ($0 == "__ROTATE__") { n = split(extra, a, ";");
            for (i = 1; i <= n; i++) if (a[i] != "") print a[i] } else print }' \
        "$SPT_TEMPLATE" \
      | sed -e "s|__INFILE__|$infile|g" -e "s|__OUTFILE__|$png|" > "$spt"
    [ "$best" = "drop" ] && sed -i '/^rotate best/d' "$spt"
    jmol -ionx -s "$spt" >/dev/null 2>&1
    rm -f "$spt"
    [ -s "$png" ] && echo "wrote $png" || { echo "error: no image for $infile" >&2; exit 1; }
}

render pp_casscf_neb_start_aligned.xyz pp_casscf_neb_start_structure.png keep \
       "rotate y 180;rotate x 90"
render pp_casscf_neb_end_aligned.xyz   pp_casscf_neb_end_structure.png   drop ""

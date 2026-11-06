#!/usr/bin/env bash
set -euo pipefail

cd /code

export THZ_DATA_ROOT="${THZ_DATA_ROOT:-/data}"
export THZ_RESULTS_ROOT="${THZ_RESULTS_ROOT:-/results}"
export THZ_SCRATCH_ROOT="${THZ_SCRATCH_ROOT:-/scratch}"
export THZ_MATERIAL_CACHE_ROOT="${THZ_MATERIAL_CACHE_ROOT:-${THZ_SCRATCH_ROOT}/material_properties}"

export MPLBACKEND="${MPLBACKEND:-Agg}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${THZ_SCRATCH_ROOT}/matplotlib}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${THZ_SCRATCH_ROOT}/cache}"
export PYVISTA_OFF_SCREEN="${PYVISTA_OFF_SCREEN:-true}"
export PYTHONUNBUFFERED=1

if [[ -d /opt/pyhelper ]]; then
    export PYTHONPATH="/opt${PYTHONPATH:+:${PYTHONPATH}}"
fi

# Code Ocean commonly mounts an attached data asset one directory below
# /data. Locate the dataset root from a stable required input when necessary.
data_marker="envelope/measurements/sample_data/resolution_target_sample.thzimg"
if [[ ! -f "${THZ_DATA_ROOT}/${data_marker}" ]]; then
    detected_marker="$(find /data -type f -path "*/${data_marker}" -print -quit 2>/dev/null || true)"
    if [[ -n "${detected_marker}" ]]; then
        export THZ_DATA_ROOT="${detected_marker%/${data_marker}}"
        echo "Detected THz data root: ${THZ_DATA_ROOT}"
    else
        echo "Could not locate ${data_marker} below /data." >&2
        echo "The data archive must be extracted; its contents may be nested under one Code Ocean asset directory." >&2
        echo "Available paths below /data:" >&2
        find /data -maxdepth 3 -mindepth 1 -print 2>/dev/null | sort | head -200 >&2
        exit 1
    fi
fi

mkdir -p \
    "${THZ_RESULTS_ROOT}" \
    "${THZ_MATERIAL_CACHE_ROOT}" \
    "${MPLCONFIGDIR}" \
    "${XDG_CACHE_HOME}"

run_python() {
    echo "Running $*"
    if [[ -x /opt/thz-venv/bin/python ]]; then
        /opt/thz-venv/bin/python -u "$@"
    else
        uv run --frozen --no-sync python -u "$@"
    fi
}

# Signal-envelope and noise figures.
run_python envelope.py
run_python estimate_noise.py

# Material-property caches and all material figures.
run_python analyse_material_properties.py
run_python plot_material_properties.py
run_python compare_dust_hesso_pellets_emt.py

# Four B-scan layer figures.
for measurement in coc coc_tilted ice ice_tilted; do
    export THZ_ACTIVE_MEASUREMENT="${measurement}"
    run_python layer_detection.py
done

# Eight full-3D datasets, each producing four PDF/PNG pairs.
for measurement in \
    coc_cube_low_res \
    ice_cube_tilted \
    solid_pebble_low_res_1 \
    solid_pebble_low_res_2 \
    solid_ice_pebble \
    spipa_b_pebble_low_res \
    spipa_b_pebble \
    empty_control
do
    export THZ_ACTIVE_MEASUREMENT="${measurement}"
    run_python full_3d_analysis.py
done

# Three transmission figures.
for measurement in solid_ice_pebble spipa_b_pebble frost_hr; do
    export THZ_ACTIVE_MEASUREMENT="${measurement}"
    run_python transmission_pebble.py
done

unset THZ_ACTIVE_MEASUREMENT

# Confirm that every reference result filename was recreated. Code Ocean may
# add its own files to /results, so unrelated platform files are ignored.
(
    cd "${THZ_DATA_ROOT}/plots"
    find . -type f ! -name '.DS_Store' | sort
) > "${THZ_SCRATCH_ROOT}/reference-manifest.txt"

reference_count="$(wc -l < "${THZ_SCRATCH_ROOT}/reference-manifest.txt" | tr -d ' ')"
if [[ "${reference_count}" != "155" ]]; then
    echo "Expected a 155-file reference manifest, found ${reference_count}." >&2
    exit 1
fi

missing=0
while IFS= read -r relative_path; do
    relative_path="${relative_path#./}"
    if [[ ! -f "${THZ_RESULTS_ROOT}/${relative_path}" ]]; then
        echo "Missing generated result: ${relative_path}" >&2
        missing=1
    fi
done < "${THZ_SCRATCH_ROOT}/reference-manifest.txt"

if [[ "${missing}" != "0" ]]; then
    exit 1
fi

echo "Successfully recreated all ${reference_count} reference result files."

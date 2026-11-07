#!/usr/bin/env bash
set -e

cd /code

export MPLBACKEND=Agg
export MPLCONFIGDIR=/scratch/matplotlib
export XDG_CACHE_HOME=/scratch/cache
export PYVISTA_OFF_SCREEN=true

mkdir -p /scratch/matplotlib /scratch/cache /results
ln -sfn /data /code/data
ln -sfn /results /code/results

# Spectroscopy analysis writes processed arrays, while Code Ocean mounts /data read-only.
if [[ ! -d /scratch/spectroscopy ]]; then
    cp -a /data/spectroscopy /scratch/spectroscopy
fi

uv run python envelope.py /data/pebbles/resolution_target_sample.thzimg
uv run python estimate_noise.py /data/pebbles/measurements/DUST_surface_res_target_3d_4/data_image_rfl

uv run python analyse_material_properties.py /scratch/spectroscopy
uv run python plot_material_properties.py /scratch/spectroscopy
uv run python compare_dust_hesso_pellets_emt.py /scratch/spectroscopy

uv run python layer_detection.py /data/pebbles/measurements/COC_PEBBLE_CoDA_T3_dust_low_res_vac_4/data_image_rfl
uv run python layer_detection.py /data/pebbles/measurements/COC_PEBBLE_CoDA_T3_dust_low_res_vac_5/data_image_rfl
uv run python layer_detection.py /data/pebbles/measurements/ICE_PEBBLE_CoDA_T3_dust_low_res_vac_5/data_image_rfl
uv run python layer_detection.py /data/pebbles/measurements/ICE_PEBBLE_CoDA_T3_dust_low_res_vac_6_small_fov_slow/data_image_rfl

uv run python full_3d_analysis.py /data/pebbles/measurements/COC_PEBBLE_CoDA_T3_dust_low_res_vac_4/data_image_rfl
uv run python full_3d_analysis.py /data/pebbles/measurements/ICE_PEBBLE_CoDA_T3_dust_low_res_vac_6_small_fov_slow/data_image_rfl
uv run python full_3d_analysis.py /data/pebbles/measurements/ICE_PEBBLE_SPHERE_CoDA_T3_dust_low_res_vac_7_fast_focus_on_sohere_top/data_image_rfl
uv run python full_3d_analysis.py /data/pebbles/measurements/ICE_PEBBLE_SPHERE_CoDA_T3_dust_low_res_vac_9_fast_focus_on_sohere_top/data_image_rfl
uv run python full_3d_analysis.py /data/pebbles/measurements/ICE_PEBBLE_SPHERE_CoDA_T3_dust_high_res_vac_9_fast_focus_on_sohere_top/data_image_rfl
uv run python full_3d_analysis.py /data/pebbles/measurements/ICE_PEBBLE_SPHERE_CoDA_T3_dust_low_res_vac_10_fast_focus_on_sphere_top/data_image_rfl
uv run python full_3d_analysis.py /data/pebbles/measurements/ICE_PEBBLE_SPHERE_CoDA_T3_dust_high_res_vac_10_fast_focus_on_sphere_top/data_image_rfl
uv run python full_3d_analysis.py /data/pebbles/measurements/CONTROL_CoDA_T3_dust_low_res_vac_11_fast_focus_on_sphere_top/data_image_rfl

uv run python transmission_pebble.py /data/pebbles/measurements/ICE_PEBBLE_CoDA_T3_dust_low_res_vac_trans_5/data_image
uv run python transmission_pebble.py /data/pebbles/measurements/ICE_PEBBLE_CoDA_T3_dust_low_res_vac_trans_4/data_image
uv run python transmission_pebble.py /data/pebbles/measurements/porosity_august2_2026_hr_focused_frost_10.0mm_7.2g/data_image

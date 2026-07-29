#!/bin/bash
# Record GelFoot topics. Press Ctrl+C to stop.

OUTPUT="gelfoot_$(date +%Y%m%d_%H%M%S)"

rosbag record -O "$OUTPUT" \
    /endcap_0/image/cropped /endcap_0/image/shear /endcap_0/image/divergence /endcap_0/array/shear_vector /endcap_0/wrench/wrench_pred /endcap_0/wrench/force_norm \
    /endcap_1/image/cropped /endcap_1/image/shear /endcap_1/image/divergence /endcap_1/array/shear_vector /endcap_1/wrench/wrench_pred /endcap_1/wrench/force_norm \
    /endcap_2/image/cropped /endcap_2/image/shear /endcap_2/image/divergence /endcap_2/array/shear_vector /endcap_2/wrench/wrench_pred /endcap_2/wrench/force_norm \
    /endcap_3/image/cropped /endcap_3/image/shear /endcap_3/image/divergence /endcap_3/array/shear_vector /endcap_3/wrench/wrench_pred /endcap_3/wrench/force_norm \
    /endcap_4/image/cropped /endcap_4/image/shear /endcap_4/image/divergence /endcap_4/array/shear_vector /endcap_4/wrench/wrench_pred /endcap_4/wrench/force_norm \
    /endcap_5/image/cropped /endcap_5/image/shear /endcap_5/image/divergence /endcap_5/array/shear_vector /endcap_5/wrench/wrench_pred /endcap_5/wrench/force_norm \
    /clock /rosout

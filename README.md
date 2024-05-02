Tips
## スレッドセーフ用設定値
```
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_MAX_THREADS=1
export OPENBLAS_MAIN_FREE=1
export OPENCV_FOR_THREADS_NUM=1
export OPENCV_FFMPEG_THREADS=1
```

## mbind failed: Function not implemented がうるさい
OpenVINOの次回アプデで解決予定。https://github.com/openvinotoolkit/openvino/pull/23874/files
**未解決**
```/etc/security/capability.conf```に下記を記載
```
cap_sys_nice=ep
```

## Midas Smallしかonnxconvertできない
2.4以降(nightly)を入れればDPT_SwinV2_T_256は行ける
timmのバージョンに注意
Tips

## mbind failed: Function not implemented がうるさい
**未解決**
```/etc/security/capability.conf```に下記を記載
```
cap_sys_nice=ep
```

## Midas Smallしかonnxconvertできない
2.4以降(nightly)を入れればDPT_SwinV2_T_256は行ける
timmのバージョンに注意
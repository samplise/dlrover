# bad sync for megatron

In this dir run

## prepare Megatron code

```
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
git checkout core_r0.9.0
git am < ../0001-feat-exp-add-bad-sync.patch
```

## run

After apply patch for megatron, ref to `Megatron-LM/BAD_SYNC.md`

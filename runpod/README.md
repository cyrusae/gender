# Running on RunPod

What the A40 trial (2026-10-06, `docs/decisions.md`) established:
- GPU time goes on **running models**: Phase 0 scoring (batched), activation extraction, and
  later steering. Analysis is CPU-only, so pods run `--extract-only` and analysis happens on the Mac.
- The **global volume** mounts at `/workspace` as GeeseFS (object storage). It's good for large
  files (~200 MB/s write, ~350 MB/s read) and bad for many small files and symlinks. So code, the
  Python environment and the Hugging Face cache live on the pod's **local disk** (`/root`); models
  worth keeping and all results are copied to `/workspace`.
- Your SSH public key is registered on the account, so pods created from now on get SSH. A pod
  created *before* the key was registered won't.

## Steps

1. **Bundle the committed code** (on the Mac): `bash runpod/bundle.sh`. It writes `/tmp/gender.tgz`
   with `BUNDLE_COMMIT.txt`, so pod results record their commit (the pod has no git history).
2. **Create a pod**: Runpod PyTorch 2.8.0 template, host CUDA ≥ 13.0 (our PyTorch build needs it),
   container disk ≥ 30 GB (more for big models), global volume at `/workspace`, port 22/tcp.
3. **Copy and set up** (`<port>`, `<ip>` from the pod's SSH details):
   ```sh
   scp -P <port> /tmp/gender.tgz root@<ip>:/root/
   ssh -p <port> root@<ip> 'mkdir -p /root/gender && cd /root/gender && tar xzf ../gender.tgz && bash runpod/setup.sh'
   ```
4. **Run** detached, and watch:
   ```sh
   ssh -p <port> root@<ip> 'cd /root/gender && nohup bash runpod/trial.sh > /dev/null 2>&1 < /dev/null &'
   ssh -p <port> root@<ip> 'tail -f /root/gender/logs/trial.log'
   ```
5. **Save results** to the volume and download them, then **terminate the pod**.

Gated models (Gemma) need a Hugging Face token: pass it as a pod-only secret/env var
(`HF_TOKEN`), not saved on the shared volume.

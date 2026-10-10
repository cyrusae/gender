The biggest speedups on rented GPUs almost always come from how you feed the GPU and what you move off it, not from the language the loop is written in. Native code is the last step, and only for the CPU-side slice of the work. So the first job is figuring out which side is slow.

**Part 1: Find out which side you're on**

Run these two measurements before reading further.

First, time the model call by itself. GPUs run asynchronously: when Python asks for a matrix multiply, the work gets queued and Python moves on. A naive timer measures how fast Python queued things, not how long they took. Call `torch.cuda.synchronize()` before reading the clock on each side. Then compare the time spent in the forward pass against the wall-clock time for the whole loop. If the forward pass is most of the time, you're model-bound. If it's a small slice, the time is going to Python, I/O, or moving data.

Second, watch the GPU while the job runs. In a second terminal, run `nvidia-smi dmon -s u` to print utilization every second. "Utilization" only means the fraction of time some kernel (a GPU function) was running, not how full the chip is. A high number means the GPU isn't idle, not that it's used efficiently.

Reading the results:
- GPU utilization high and steady, forward pass dominates: model-bound. The levers are in Part 3.
- GPU dropping to idle in bursts, one CPU core pegged at 100%: Python-bound. The GPU is waiting on your loop.
- Doubling batch size roughly halves time per example: you were overhead-bound, where fixed per-call cost dominated. Batching is the fix.

To test the batching question directly, run the same job at batch sizes 1, 8, and 32 and watch time per example.

**Part 2: Find the Python hotspot without touching the code**

`py-spy` is a sampling profiler written in Rust that attaches to a running process from outside. `py-spy top --pid <PID>` shows live hot functions, and `py-spy record -o profile.svg --pid <PID>` produces a flame graph. Some container setups refuse the ptrace permission it needs, so a permissions error there is likely the cause. For the GPU side, PyTorch's built-in profiler (`torch.profiler.profile` with CPU and CUDA activities) shows which operations took time and where the GPU sat idle. Look for floods of tiny kernels and points where the CPU blocks on the GPU.

**Part 3: The usual suspects in mech interp loops**

These are caused by your code, and they're usually cheap to fix:

- **Syncs inside the hot loop.** `.item()`, `.cpu()`, printing a tensor, or branching on a tensor's value each force the CPU to wait for the GPU. Do them once per batch, not once per prompt.
- **Batch size of 1.** This is the most common mistake in these setups.
- **Hooks that copy everything to CPU.** If your activation-caching hook saves the full residual stream at every layer and position, you're pushing a lot across PCIe (the bus between CPU and GPU). Hook libraries like TransformerLens let you filter by name (`names_filter`). Cache only what you'll analyze, and keep it on the GPU until the batch finishes.
- **Generating when you could score.** For probing grammatical gender bleed, you probably need activations and logit comparisons on fixed text, which is one forward pass. `model.generate` runs a Python loop of decoding steps, each with overhead. If you only generate to read off next-token probabilities, skip generation.
- **Many small file writes.** Accumulate results and write them in chunks, such as safetensors or NumPy shards.
- **Reloading the model every run.** Loading weights can take minutes. Keep one process or notebook kernel alive across experiments.

These are caused by the model, and coding won't remove them:

- **Decoding is memory-bound.** Each generated token reads the whole KV cache (the stored keys and values for past tokens). Batching is the main lever.
- **Attention cost grows with the square of sequence length.** Long prompts cost more than you'd expect.
- **Precision.** bf16 (bfloat16, a 16-bit float with fp32's range) moves half the bytes of fp32. Check that your hooks and the rest of the pipeline stay in bf16 consistently.
- **Attention kernels vs. attention weights.** Fast kernels like PyTorch's SDPA and FlashAttention don't return attention matrices. Experiments that need attention patterns have to take the slower eager path, so know which experiments need which.
- **Quantization changes what you're measuring.** int8, int4, and fp8 speed up inference, but they alter the activations you're probing. For a study about internal representations, that's a validity cost, not just a speed trade. Quantize only after checking that the features survive.

**Part 4: Rented-box checks**

`nproc` and `htop` tell you how many CPU cores you actually got and whether one is pegged. `free -h` shows RAM. Know where your weights and datasets live: network-mounted volumes often read much slower than local disk, so copy hot files locally before a run. `nvidia-smi -q -d PERFORMANCE` reports throttling if the card is power-capped or running hot.

**Part 5: Native code, when it helps**

Rust or C speeds up CPU-side Python work. It won't make a matmul faster, since those already run as optimized CUDA kernels. Native code pays off when the profile shows time spent tokenizing or filtering a large corpus, parsing JSON, running metrics in Python loops, or munging files.

Here are the options, cheapest first:

1. **Vectorize with NumPy or torch.** Replacing a Python loop with array operations is often a large win, with no new language.
2. **Use libraries that are already native underneath.** Hugging Face `tokenizers` and `safetensors` are written in Rust. `polars` is a Rust-backed dataframe library that can replace pandas for big tabular work. You may already be running Rust without noticing.
3. **Numba.** The `@njit` decorator compiles numeric Python loops to machine code. Same language, big speedups on loops that don't vectorize.
4. **PyO3 with maturin.** PyO3 lets you write a `#[pyfunction]` in Rust and call it from Python. Maturin builds it: `maturin new --bindings pyo3`, then `maturin develop --release`. You can pass NumPy arrays across without copying (via the rust-numpy crate), and parallelize with rayon while releasing the GIL (the global lock that stops Python threads from running Python code in parallel) using `py.allow_threads`. If you end up using C++ for coursework, pybind11 is the equivalent.
5. **Subprocess to a compiled binary.** Good for large batch jobs: write input to Parquet or JSON lines, run the binary, read its output. The cost is process startup and serialization, so it's fine for big jobs and bad for millions of tiny calls.
6. **ctypes or cffi against a C shared library.** It works, but memory management is entirely your responsibility.

The main pitfall is the cost of crossing the Python-to-native boundary. A million tiny calls can erase the gains, so cross once with large contiguous buffers rather than once per item. Also, Python threads won't speed up pure Python code because of the GIL, so use multiprocessing or release the GIL on the native side.

Your next step is to run the timing split from Part 1 on your actual loop. Then tell me whether it's calling `generate()` or running forward passes, and what the numbers were. From that I can tell you which side you're on and what to try first.
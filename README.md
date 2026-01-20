# SGLang Server Configuration

This directory contains configuration files for running SGLang, a fast serving framework for large language models. SGLang is configured to run with the same models as the vLLM setup.

## Available Models

- **Llama 3.2 3B Instruct** (unsloth/Llama-3.2-3B-Instruct)
  - 3B parameters
  - Native 128K context window
  - Tool calling support
  - VRAM usage: ~3GB

- **Qwen 2.5 7B Instruct AWQ** (Qwen/Qwen2.5-7B-Instruct-AWQ)
  - 7B parameters (4-bit quantized)
  - Native 32K context window
  - Excellent tool calling capabilities
  - VRAM usage: ~4GB

## Quick Start

### Installation

Run the setup script to install SGLang and all dependencies:

```bash
sudo bash setup-sglang.sh
```

This will:
- Configure system to prevent sleep (for remote access)
- Install Python and system dependencies
- Install CUDA 12.8 (if not already installed)
- Create a Python virtual environment at `~/sglang-env`
- Install SGLang with FlashInfer support
- Create a systemd service for auto-start

### Starting the Server

#### Interactive Mode
```bash
./start-sglang.sh
```

Choose from the menu:
1. Llama 3.2 3B Instruct
2. Qwen 2.5 7B Instruct AWQ

#### Direct Launch
```bash
./start-sglang.sh llama    # Launch Llama 3.2 3B
./start-sglang.sh qwen      # Launch Qwen 2.5 7B
```

#### Using YAML Configuration
```bash
source ~/sglang-env/bin/activate
python -m sglang.launch_server --config llama-3.2-3b.yaml
# or
python -m sglang.launch_server --config qwen2.5-7b.yaml
```

### Testing the Server

```bash
./test-sglang.sh
```

This will run several tests:
1. Health check
2. Model info retrieval
3. Completion API test
4. Chat completion API test

### Using as a System Service

```bash
# Start the service
sudo systemctl start sglang

# Enable auto-start on boot
sudo systemctl enable sglang

# Check status
sudo systemctl status sglang

# View logs
sudo journalctl -u sglang -f
```

## Configuration Comparison: vLLM vs SGLang

| Feature | vLLM | SGLang |
|---------|------|--------|
| **Command** | `vllm serve` | `python -m sglang.launch_server` |
| **Max Context** | `--max-model-len` | `--context-length` |
| **Max Concurrent** | `--max-num-seqs` | `--max-running-requests` |
| **GPU Memory** | `--gpu-memory-utilization` | `--mem-fraction-static` |
| **Port** | `--port` | `--port` |
| **Host** | *(default 127.0.0.1)* | `--host` |
| **Data Type** | *(auto)* | `--dtype half` |
| **Attention Backend** | *(auto)* | `--attention-backend flashinfer` |

### Current Configuration

Both vLLM and SGLang are configured with equivalent settings:

- **Context Length**: 16384 tokens (limited from native 128K/32K for memory efficiency)
- **Max Concurrent Requests**: 16
- **GPU Memory Utilization**: 90%
- **Port**: 8000
- **Data Type**: FP16 (half precision)

## API Usage

SGLang provides an OpenAI-compatible API:

### Chat Completion

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is the capital of France?"}
    ],
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

### Text Completion

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "prompt": "The capital of France is",
    "max_tokens": 10,
    "temperature": 0.0
  }'
```

### Python Example

```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY"  # SGLang doesn't require an API key
)

response = client.chat.completions.create(
    model="default",
    messages=[
        {"role": "user", "content": "Hello!"}
    ],
    max_tokens=100
)

print(response.choices[0].message.content)
```

## Advanced Configuration

### Custom Context Length

To use the full context (requires more VRAM):

```bash
python -m sglang.launch_server \
  --model-path unsloth/Llama-3.2-3B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --context-length 131072 \
  --max-running-requests 8 \
  --mem-fraction-static 0.95
```

For Qwen 2.5 7B (32K context max):

```bash
python -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-7B-Instruct-AWQ \
  --host 0.0.0.0 \
  --port 8000 \
  --context-length 32768 \
  --max-running-requests 4 \
  --mem-fraction-static 0.90
```

### Enable Metrics and Logging

Add these flags to your launch command:

```bash
--enable-metrics \
--log-requests \
--log-level info
```

Or uncomment them in the YAML config files.

### Multi-GPU Setup

For tensor parallelism across multiple GPUs:

```bash
python -m sglang.launch_server \
  --model-path unsloth/Llama-3.2-3B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 2
```

For Qwen 2.5 7B with AWQ quantization:

```bash
python -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-7B-Instruct-AWQ \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 2
```

## Performance Tuning

### FlashInfer Attention Backend

SGLang uses FlashInfer for optimized attention computation:
- Faster inference than standard attention
- Lower memory usage
- Better batching efficiency

### Memory Optimization

Adjust `mem-fraction-static` based on available VRAM:
- `0.70` - Conservative (leaves room for other processes)
- `0.85` - Balanced (default)
- `0.95` - Aggressive (maximum performance)

### Batch Size Tuning

Adjust `max-running-requests` based on your workload:
- Lower values: Better latency per request
- Higher values: Better throughput for many concurrent requests

## Troubleshooting

### Server Won't Start

1. Check if port 8000 is already in use:
   ```bash
   sudo netstat -tulpn | grep 8000
   ```

2. Check GPU availability:
   ```bash
   nvidia-smi
   ```

3. View detailed logs:
   ```bash
   sudo journalctl -u sglang -n 100
   ```

### Out of Memory Errors

Reduce memory usage:
- Lower `context-length` (e.g., 8192)
- Reduce `max-running-requests` (e.g., 8)
- Lower `mem-fraction-static` (e.g., 0.80)

### Slow Response Times

Increase performance:
- Stop display manager: `sudo systemctl stop gdm3`
- Increase `mem-fraction-static`
- Ensure FlashInfer is installed correctly

## Files in This Directory

- `setup-sglang.sh` - Installation script
- `start-sglang.sh` - Interactive server launcher
- `test-sglang.sh` - Server testing script
- `llama-3.2-3b.yaml` - Llama 3.2 3B configuration
- `qwen2.5-7b.yaml` - Qwen 2.5 7B configuration
- `README.md` - This file

## Additional Resources

- [SGLang Documentation](https://github.com/sgl-project/sglang)
- [SGLang GitHub](https://github.com/sgl-project/sglang)
- [FlashInfer Documentation](https://flashinfer.ai/)

## License

This configuration follows the same license as the underlying models:
- Llama 3.2: Llama 3.2 Community License
- Qwen 2.5: Apache 2.0 License

from __future__ import annotations

import argparse
import subprocess


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='Qwen/Qwen2.5-Coder-32B-Instruct')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--adapter', default=None)
    parser.add_argument('--gpu-memory-utilization', default='0.80')
    parser.add_argument('--served-model-name', default=None)
    args = parser.parse_args()

    cmd = [
        'python', '-m', 'vllm.entrypoints.openai.api_server',
        '--model', args.model,
        '--port', str(args.port),
        '--gpu-memory-utilization', str(args.gpu_memory_utilization),
    ]
    if args.served_model_name:
        cmd.extend(['--served-model-name', args.served_model_name])
    if args.adapter:
        cmd.extend(['--enable-lora', '--lora-modules', f'agent_adapter={args.adapter}'])
    print('Starting vLLM:', ' '.join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == '__main__':
    main()

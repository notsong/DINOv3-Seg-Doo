"""Bounded synthetic decoder benchmark; no backbone or accuracy evaluation.

Run from repository root: python tests/benchmark_decoder.py
"""
import gc
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch import nn

from models.decoder_v3 import DecoderV3
from models.detail_encoder import DetailEncoder


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_num_threads(2)
    report = {'device': torch.cuda.get_device_name() if device == 'cuda' else 'cpu',
              'input_size': [1, 3, 256, 256], 'dtype': 'float16' if device == 'cuda' else 'float32',
              'scope': 'decoder + optional detail CNN only; excludes DINOv3 and preprocessing', 'runs': {}}
    for enabled in (False, True):
        decoder = DecoderV3(2, use_detail_branch=enabled).eval().to(device)
        detail = DetailEncoder().eval().to(device) if enabled else None
        if device == 'cuda':
            decoder.half()
            if detail is not None:
                detail.half()
        dtype = next(decoder.parameters()).dtype
        image = torch.randn(1, 3, 256, 256, device=device, dtype=dtype)
        feats = {int(layer): torch.randn(1, 768, 16, 16, device=device, dtype=dtype)
                 for layer in decoder.fusion.cbam}
        conv_macs = {}
        handles = []

        def hook_for(name):
            def hook(module, inputs, output):
                conv_macs[name] = conv_macs.get(name, 0) + output.numel() * (module.in_channels // module.groups) * module.kernel_size[0] * module.kernel_size[1]
            return hook

        for prefix, model in (('decoder', decoder), ('detail', detail)):
            if model is not None:
                for name, module in model.named_modules():
                    if isinstance(module, nn.Conv2d):
                        handles.append(module.register_forward_hook(hook_for(prefix + '.' + name)))

        def forward():
            details = detail(image) if detail is not None else None
            return decoder(feats, output_size=(256, 256), details=details)

        with torch.inference_mode():
            forward()
            for handle in handles:
                handle.remove()
            for _ in range(3):
                forward()
            if device == 'cuda':
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            timings = []
            for _ in range(10):
                start = time.perf_counter()
                outputs = forward()
                if device == 'cuda':
                    torch.cuda.synchronize()
                timings.append((time.perf_counter() - start) * 1000)
                del outputs
        groups = {}
        for name, macs in conv_macs.items():
            group = '.'.join(name.split('.')[:2]) if name.startswith('decoder.') else 'detail'
            groups[group] = groups.get(group, 0) + macs
        report['runs']['detail' if enabled else 'baseline'] = {
            'parameters': sum(p.numel() for p in decoder.parameters()) + (sum(p.numel() for p in detail.parameters()) if detail else 0),
            'median_ms': sorted(timings)[len(timings) // 2],
            'peak_allocated_mib': torch.cuda.max_memory_allocated() / 1024**2 if device == 'cuda' else None,
            'conv_macs_by_group': groups}
        del decoder, detail, image, feats
        gc.collect()
        if device == 'cuda':
            torch.cuda.empty_cache()
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

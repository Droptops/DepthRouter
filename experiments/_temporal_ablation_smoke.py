from __future__ import annotations

from hf_temporal_tangent_distilled import load_texts


def main() -> None:
    texts = load_texts(None)
    print(f'temporal ablation corpus size: {len(texts)}')


if __name__ == '__main__':
    main()

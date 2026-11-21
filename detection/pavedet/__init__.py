"""pavedet: pipeline de detecção de danos em pavimentos com YOLO (Ultralytics).

Módulos:
    config     - leitura/mescla de YAML e overrides de linha de comando
    io_utils   - imagens (inclui HEIC/EXIF), rótulos YOLO, hashes perceptuais
    splitting  - agrupamento de quase-duplicatas e divisão estratificada por grupo
    env        - informações do ambiente (device, versões, git)
    tracking   - registro de execuções (parâmetros, métricas, pesos, tempos)
    metrics    - extração de P, R, F1, mAP por classe a partir da Ultralytics
    training   - montagem dos argumentos e execução de um treino
    plots      - gráficos comparativos
"""

__version__ = "1.0.0"

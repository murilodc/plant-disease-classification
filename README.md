# Detecção de Doenças em Folhas de Plantas

Projeto desenvolvido para o Trabalho de Conclusão de Curso em Ciência da Computação.

O trabalho investiga a classificação de doenças em folhas de plantas utilizando redes neurais convolucionais e técnicas de transferência de aprendizado sobre o dataset PlantVillage.
## Repetições com seeds 123 e 456

O notebook `notebooks/07_repeticoes_individuais.ipynb` prepara **um modelo e uma
semente por vez**, reutilizando `src/train_baseline.py`, `src/train_resnet50.py`,
`src/train_densenet201.py` e `src/train_efficientnet_b0.py`.

O protocolo permanece: batch 32; até 30 épocas; Adam; learning rate 0,001 na
baseline e 0,0001 nos modelos pré-treinados; early stopping por validation loss
(patience 5, min_delta 0,0001); checkpoint de menor validation loss restaurado
antes das métricas; imagens 224×224 e os mesmos transforms de cada script.
`num_workers` é copiado da referência do modelo. A nova seed afeta somente RNG,
augmentations e shuffle; o arquivo `results/plantvillage_metadata_split.csv` é
reutilizado sem gerar divisões. Os treinamentos criam apenas loaders de treino
e validação e nunca avaliam o teste.

Os resultados existentes permanecem em `results/<modelo>/` (seed 42).
Novas execuções ficam em `results/repetitions/seed_123/<modelo>/` e
`results/repetitions/seed_456/<modelo>/`. Os nomes dos checkpoints e artefatos
internos continuam iguais. O executor reserva cada diretório antes do lançamento,
registra um manifesto com seed, comando, hash SHA-256 do CSV e da metadata de
referência, e bloqueia qualquer nova execução no mesmo destino. Execuções
interrompidas ficam preservadas com status `failed`; para repetir, arquive
manualmente esse diretório antes, mantendo os arquivos anteriores. Os scripts
de treino também recusam destinos que já contenham artefatos.

### Primeira repetição no Colab

1. Abra `notebooks/07_repeticoes_individuais.ipynb` no Colab e selecione GPU L4.
2. Monte o Drive pela primeira célula. Ajuste `PROJECT_ROOT` para a pasta com
   **src atualizado, CSV original e os quatro resultados L4 completos da seed 42**.
   Não execute os notebooks de exploração/divisão para preparar as repetições.
3. Use o mesmo ambiente PyTorch, torchvision e CUDA registrado em
   `results/<modelo>/experiment_metadata.json`. O executor confere essas versões
   e GPU L4 antes de lançar o processo. O notebook não instala dependências.
4. Mantenha `MODEL = 'baseline_cnn'` e `SEED = 123`. Ajuste `IMAGE_ROOT` para as
   imagens locais. Se precisar extrair, ajuste `ZIP_PATH` para o ZIP original e
   marque `EXTRACT_IMAGES = True`: só os membros de treino/validação são extraídos.
5. Execute a preparação e confira o comando, destino e hash impressos.
   Inicialmente `EXECUTE_TRAINING = False`, portanto nenhum treino é iniciado.
6. Para iniciar a primeira repetição, altere para `EXECUTE_TRAINING = True`,
   reexecute a célula de seleção e depois a célula de treinamento. Os artefatos
   serão persistidos em `results/repetitions/seed_123/baseline_cnn/` no Drive.
7. Depois selecione os demais modelos individualmente com seed 123; faça o
   mesmo com seed 456. Execute a célula final para atualizar a comparação.

**Referência local:** os arquivos presentes neste checkout podem ser anteriores
às execuções Colab informadas: por exemplo, a metadata local da baseline registra
RTX 5060 e `num_workers=0`. Esses arquivos não foram corrigidos nem substituídos.
Use os artefatos L4 verdadeiros no Drive; o executor bloqueia referências
incompletas/inválidas e GPUs diferentes. O hash registra o CSV atual: como os
artefatos antigos não registram seu hash, a correspondência histórica com a seed
42 depende de fornecer o CSV original preservado. Entre novas repetições, hashes
diferentes são bloqueados.

Também é possível preparar pela CLI (não inicia treinamento):

```bash
python src/run_repetition.py --model baseline_cnn --seed 123 --image-root /content/plantvillage_color
```

Acrescente `--execute` somente para iniciar a execução. Use `--project-root` se
os scripts e resultados estiverem em outra pasta.

### Comparação das repetições

```bash
python src/compare_repetitions.py --results-root results
```

A mesma função está na última célula do notebook 07. A comparação antiga no
notebook 06 e `compare_models.py` continua disponível para a seed 42.
A nova comparação usa as três seeds planejadas e salva em
`results/repetitions_comparison/`:

- `validation_individual.csv`: uma linha por modelo/seed, incluindo status,
  métricas, tempos e condições de execução (12 linhas planejadas).
- `validation_aggregate.csv`: média e desvio padrão **amostral, ddof=1** das
  sete métricas de classificação, loss, épocas e tempos de cada modelo;
  inclui `n_runs`, seeds válidas e status `partial`/`complete`.
- `validation_by_class_individual.csv` e `validation_by_class_aggregate.csv`:
  métricas individuais e estatísticas por classe, com contagens.
- `histories_individual.csv`: curvas de cada execução, preservando suas épocas
  reais, sem alinhar ou interpolar early stopping entre seeds.
- `comparison_warnings.txt`: pendências, inconsistências e diferenças de
  hardware, workers ou ambiente PyTorch/CUDA, inclusive entre sementes.

Somente execuções completas, com seed correta e artefatos válidos entram nas
médias; as novas também exigem manifesto concluído e hashes correspondentes.
Com uma execução, o desvio padrão fica NaN (indefinido); sem execução válida,
média e desvio ficam NaN. Execuções ausentes ou inválidas não viram zeros.
Médias parciais indicam o número real de repetições e não representam o estudo
completo. Não interprete tempos com avisos de condições diferentes como uma
comparação controlada. Não há escolha de seed vencedora nem uso de teste.

### Testes

```bash
python -m unittest discover -s tests -v
```

Os testes usam exclusivamente artefatos sintéticos em pastas temporárias;
verificam médias/desvios, contagens, ausência/invalidação, hashes, preservação de
arquivos, bloqueios de sobrescrita, seleção do notebook e loaders sem teste.
O lançamento de processo é simulado nos testes do executor; nenhum treinamento
é realizado. O notebook é validado e suas células de preparação, bloqueio de
treino e comparação são executadas sobre os dados sintéticos.

Neste checkout, `.venv/Scripts/python.exe` resolve para Python 3.12, mas as
extensões instaladas são CPython 3.14. A suíte foi validada com o Python 3.14
instalado e as bibliotecas existentes, sem modificar a `.venv`:

```powershell
py -3.14 -c "import sys; sys.path.insert(0, r'.venv/Lib/site-packages'); import unittest; result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.discover('tests')); sys.exit(not result.wasSuccessful())"
```

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

### Colab: `No module named run_repetition`

`run_repetition` é `src/run_repetition.py`, não um pacote do pip. Abrir o
notebook pelo GitHub não atualiza a cópia dos scripts no Drive. Confira o
`PROJECT_ROOT` impresso na célula de caminhos. Em um clone Git, atualize essa
pasta antes de reexecutar as células de caminhos, seleção e preparação:

```python
import subprocess
subprocess.run(['git', '-C', str(PROJECT_ROOT), 'pull', '--ff-only'], check=True)
print(PROJECT_ROOT / 'src/run_repetition.py')
assert (PROJECT_ROOT / 'src/run_repetition.py').is_file()
```

Se a pasta não for um clone, copie a pasta `src/` atualizada do repositório
para `PROJECT_ROOT/src/`. Preserve o CSV, dados e resultados existentes. Se o
Git bloquear o pull por alterações locais, preserve essas alterações antes de
sincronizar; não use reset ou limpeza forçada. A atualização não inicia treino.

### Avaliação final no teste — notebook 08

`notebooks/08_avaliacao_teste.ipynb` usa `src/evaluate_test.py` para avaliar os
12 checkpoints previamente escolhidos na validação (quatro modelos, seeds
42/123/456). Não executa treinamento nem seleciona modelos, hiperparâmetros,
épocas ou seeds com base no teste. **A avaliação real vem desativada**; executar
todas as células nessa configuração não abre imagens nem gera resultados de teste.

No Google Colab:

1. Sincronize `src/` e abra o notebook 08. Selecione um runtime com GPU e monte
   o Drive na primeira célula. Para comparar tempos controlados, preserve o
   ambiente/GPU L4 dos experimentos; métricas de classificação não exigem L4.
2. Ajuste `PROJECT_ROOT` para a pasta com os artefatos usados no notebook 07.
   Preserve `results/plantvillage_metadata_split.csv` original, os diretórios
   `results/<modelo>/` da seed 42 e
   `results/repetitions/seed_<123 ou 456>/<modelo>/` com seus manifestos.
   A cópia local deste checkout contém apenas a baseline original; os resultados
   completos devem ser fornecidos pelo Drive. Não recrie a divisão.
3. Execute a configuração e a preparação mantendo
   `EXECUTE_TEST_EVALUATION = False`. A preparação exige 12 execuções completas,
   confere hashes, metadados, seeds, mapas de classes, isolamento de `leaf_id`,
   contagens 38.008/8.172/8.125 e restaura todos os checkpoints na CPU com
   carregamento estrito. Nenhuma imagem é aberta nessa etapa.
4. Somente quando for realizar a avaliação final, ajuste `IMAGE_ROOT` para as
   imagens extraídas. Se necessário, ajuste `ZIP_PATH` e habilite
   `EXTRACT_TEST_IMAGES`: a célula extrai apenas os caminhos de teste do CSV,
   com ambas as chaves habilitadas. Não sobrescreve imagens locais existentes.
5. Altere deliberadamente `EXECUTE_TEST_EVALUATION = True` e execute a partir
   da configuração. O avaliador verifica a integridade de todas as imagens,
   executa os 12 modelos em modo de inferência e salva em uma pasta exclusiva
   `results/test_evaluation/<data_UTC>_<id>/`. Checkpoints, validações e repetições
   permanecem preservados. A última célula mostra resultados individuais e
   consolidados; use somente execuções com manifesto `status=complete`.

O pré-processamento reproduz a validação existente: RGB, resize 224×224 e
normalização ImageNet, sem augmentation aleatório. Na ResNet50 reutiliza a
função específica da arquitetura; DenseNet201 e EfficientNet-B0 conferem a
compatibilidade do pipeline compartilhado. Mantém-se a geometria usada no
treinamento, inclusive a decisão registrada de não aplicar o recorte oficial
da EfficientNet-B0. Os modelos são construídos com `weights=None`; os pesos
aprendidos vêm exclusivamente dos checkpoints existentes.

São salvos acurácia, precisão/recall/F1 macro e ponderados, métricas por classe,
predições identificadas por caminho, matrizes de confusão absolutas e
normalizadas em CSV e absolutas em PNG, tempos e descrição dos transforms.
As tabelas `test_individual.csv` e `test_aggregate.csv` consolidam as sete
métricas, com média e **desvio padrão amostral (ddof=1)** entre as três seeds.
As tabelas `test_by_class_*.csv` fazem o mesmo por classe, preservando o suporte
original. Não há ranking, escolha da melhor seed ou ensemble baseado no teste.

O manifesto registra ambiente, fontes e saídas com SHA-256 e o estado
`running`/`complete`/`failed`. O CSV exato de teste, o mapa de classes e hashes
de todas as imagens são preservados junto aos resultados. Uma falha não descarta
imagens nem produz médias parciais; a pasta permanece como evidência e outra
execução exige uma nova pasta.

Limite dos artefatos legados: seed/modelo/hash do CSV não estão incorporados nos
checkpoints antigos. A identidade depende dos metadados/manifestos associados,
com conferência da arquitetura, mapas, época e métricas de validação. O vínculo
histórico da seed 42 ao CSV depende de fornecer o original preservado; os hashes
das repetições são verificados. Essa limitação é registrada, sem modificar os
artefatos antigos. Arquivos de checkpoint são carregados com `weights_only=True`.

Os testes em `tests/test_evaluate_test.py` usam somente CSVs, imagens e
checkpoints sintéticos em diretórios temporários. Cobrem o fluxo de inferência,
as 12 execuções, integridade, classes/seeds/splits, métricas conhecidas, ddof=1,
preservação de fontes, bloqueio de sobrescrita e a chave desativada do notebook.
Execute com `python -m unittest discover -s tests -v` (ou com o comando Python
3.14 acima para este ambiente local). Nenhuma avaliação real é executada pelos
testes ou pela implementação desta tarefa.

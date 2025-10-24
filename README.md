# Análise de Dados
Exercício da disciplina de Tópicos Especiais em Software

## Instruções do Exercício

Realizar análise de dados utilizando as técnicas discutidas em sala sobre não-supervisionado utilizando clusters.

## Passos

1. Entrar no site https://scop.berkeley.edu/astral/ver=2.08 e baixar a versão do banco de dados de sequências de proteínas, a base é de texto puro com header seguido de sequência. Cada classe é definida pelos primeiros elementos do header
2. Extrair os atributos de sequência segundo o que fizemos em sala, um kmer de 2X2, onde 2 são os pares de aminoácidos, e X é um skip. Fazer uma janela móvel deste kmer que extraia ausência e presença de todos os elementos das sequências e a transforme em uma matrix binária com todas as combinações possíveis
3. Projetar a matriz geradas em uma base de PCA utilizando os 300 componentes principais
4. Realizar a escolha do algoritmo de cluster rodando todos os algoritmos do sci-kit no python, calculando as métricas internas e as correlacionando com a métrica externa F1-Score
5. Variar os parâmetros para obter a melhor métrica interna e sugerir a melhor configuração de clusters

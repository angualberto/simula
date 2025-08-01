import random
from itertools import permutations
import hashlib # Para gerar a "semente" a partir da sequência
import matplotlib.pyplot as plt # type: ignore
import matplotlib.animation as animation # type: ignore
import time
import os
import os
from Bio import SeqIO

# Caminho da pasta onde estão os arquivos
PASTA = "."


def listar_fasta(pasta):
    return [f for f in os.listdir(pasta) if f.endswith(".fasta")]

def exibir_opcoes(fastas):
    print("\nArquivos disponíveis:")
    for i, nome in enumerate(fastas):
        print(f"[{i}] {nome}")
    escolha = int(input("\nEscolha o número da sequência que deseja usar: "))
    return fastas[escolha]

def carregar_sequencia(caminho):
    with open(caminho, "r") as f:
        for record in SeqIO.parse(f, "fasta"):
            print(f"\n> ID: {record.id}")
            print(f"> Descrição: {record.description}")
            return str(record.seq)

def main():
    fastas = listar_fasta(PASTA)
    if not fastas:
        print("Nenhum arquivo .fasta encontrado.")
        return
    selecionado = exibir_opcoes(fastas)
    caminho = os.path.join(PASTA, selecionado)
    sequencia = carregar_sequencia(caminho)

    print(f"\nSequência carregada com sucesso! ({len(sequencia)} bases)")
    
    # Aqui você pode chamar a simulação com IA:
    # resultados = simular_mutacoes(sequencia, ...)
    # print(resultados)

if __name__ == "__main__":
    main()

# --- CONSTANTES DO MODELO ---
A_GLC = 1664525
C_GLC = 1013904223
M_GLC = 2**32
LIMIAR_MUTACAO = 0.02

CODIGO_GENETICO_COMPLETO = {
    'UUU':'F', 'UUC':'F', 'UUA':'L', 'UUG':'L',
    'UCU':'S', 'UCC':'S', 'UCA':'S', 'UCG':'S',
    'UAU':'Y', 'UAC':'Y', 'UAA':'*', 'UAG':'*',
    'UGU':'C', 'UGC':'C', 'UGA':'*', 'UGG':'W',
    'CUU':'L', 'CUC':'L', 'CUA':'L', 'CUG':'L',
    'CCU':'P', 'CCC':'P', 'CCG':'P', 'CCT':'P',
    'CAU':'H', 'CAC':'H', 'CAA':'Q', 'CAG':'Q',
    'CGU':'R', 'CGC':'R', 'CGA':'R', 'CGG':'R',
    'AUU':'I', 'AUC':'I', 'AUA':'I', 'AUG':'M',
    'ACU':'T', 'ACC':'T', 'ACA':'T', 'ACG':'T',
    'AAU':'N', 'AAC':'N', 'AAA':'K', 'AAG':'K',
    'AGU':'S', 'AGC':'S', 'AGA':'R', 'AGG':'R',
    'GUU':'V', 'GUC':'V', 'GUA':'V', 'GUG':'V',
    'GCU':'A', 'GCC':'A', 'GCG':'A', 'GCT':'A',
    'GAC':'D', 'GAT':'D', 'GAA':'E', 'GAG':'E',
    'GGA':'G', 'GGC':'G', 'GGG':'G', 'GGT':'G',
    'TCA':'S', 'TCC':'S', 'TCG':'S', 'TCT':'S',
    'TTC':'F', 'TTT':'F', 'TTA':'L', 'TTG':'L',
    'TAC':'Y', 'TAT':'Y', 'TAA':'*', 'TAG':'*',
    'TGC':'C', 'TGT':'C', 'TGA':'*', 'TGG':'W'
}

BASES_DNA = ['A', 'T', 'C', 'G']

def carregar_sequencia_fasta(caminho_arquivo):
    try:
        with open(caminho_arquivo, 'r') as f:
            linhas = f.readlines()
        sequencia = ''.join([linha.strip() for linha in linhas if not linha.startswith(">")])
        return sequencia.upper()
    except FileNotFoundError:
        print(f"Erro: Arquivo '{caminho_arquivo}' não encontrado.")
        return None

def transcrever_para_rna(sequencia_dna):
    return sequencia_dna.replace("T", "U")

def traduzir_para_proteina(sequencia_rna):
    proteina = ''
    for i in range(0, len(sequencia_rna) - 2, 3):
        codon = sequencia_rna[i:i+3]
        aminoacido = CODIGO_GENETICO_COMPLETO.get(codon, '?')
        if aminoacido == '*':
            break
        proteina += aminoacido
    return proteina

def gerar_numero_aleatorio_glc(x_n):
    return (A_GLC * x_n + C_GLC) % M_GLC

def escolher_nova_base(original_base):
    bases_possiveis = [b for b in BASES_DNA if b != original_base]
    return random.choice(bases_possiveis)

def aplicar_mutacoes(sequencia_dna, semente_glc_inicial):
    nova_sequencia = list(sequencia_dna)
    semente_atual = semente_glc_inicial
    for i in range(len(nova_sequencia)):
        semente_atual = gerar_numero_aleatorio_glc(semente_atual)
        probabilidade = semente_atual / M_GLC
        if probabilidade < LIMIAR_MUTACAO:
            base_original = nova_sequencia[i]
            nova_base = escolher_nova_base(base_original)
            nova_sequencia[i] = nova_base
    return "".join(nova_sequencia), semente_atual

class IAClassi:
    def __init__(self):
        self.modelo = {}
        self.taxa_aprendizado = 0.01
        self.memoria = []
    def atualizar_dados(self, entrada_proteina, impacto):
        self.memoria.append((entrada_proteina, impacto))
    def treinar(self):
        if not self.memoria:
            return
        print(f"    [IAClassi] Treinando com {len(self.memoria)} novos dados.")
    def classificar(self, sequencia_proteina):
        protein_hash = int(hashlib.sha256(sequencia_proteina.encode()).hexdigest(), 16) % 1000000
        base_impact = (len(sequencia_proteina) / 100) * 0.5
        learned_impact = 0
        known_count = 0
        for p_mem, imp_mem in self.memoria:
            if p_mem == sequencia_proteina:
                learned_impact += imp_mem
                known_count += 1
        if known_count > 0:
            learned_impact /= known_count
            return (base_impact + learned_impact) / 2 + random.uniform(-0.05, 0.05)
        else:
            return base_impact + random.uniform(-0.1, 0.1)

def avaliar_impacto_biologico(proteina):
    return random.uniform(0.1, 1.0) * (len(proteina) / 50.0)

def inferir_funcao(proteina):
    return hashlib.sha256(proteina.encode()).hexdigest()

def similaridade(proteina1, proteina2):
    if proteina1 == proteina2:
        return 1.0
    min_len = min(len(proteina1), len(proteina2))
    if min_len == 0: return 0.0
    matches = sum(1 for a, b in zip(proteina1[:min_len], proteina2[:min_len]) if a == b)
    return matches / min_len

def energia_dobra(proteina):
    return random.uniform(-10.0, -1.0) + (len(proteina) / 100.0)

def remover_equivalentes(proteinas_grupo, ia_model):
    unicas = []
    for p in proteinas_grupo:
        equivalente = False
        for u in unicas:
            sim_estrutural = similaridade(p, u)
            diff_ia_impacto = abs(ia_model.classificar(p) - ia_model.classificar(u))
            if sim_estrutural > 0.95 and diff_ia_impacto < 0.05:
                equivalente = True
                break
        if not equivalente:
            unicas.append(p)
    return unicas

def selecionar_representante(proteinas_grupo_limpo, ia_model):
    if not proteinas_grupo_limpo:
        return None
    melhor_representante = proteinas_grupo_limpo[0]
    melhor_impacto = ia_model.classificar(melhor_representante)
    melhor_energia = energia_dobra(melhor_representante)
    for p in proteinas_grupo_limpo[1:]:
        impacto_atual = ia_model.classificar(p)
        energia_atual = energia_dobra(p)
        if energia_atual < melhor_energia and impacto_atual > melhor_impacto:
            melhor_representante = p
            melhor_impacto = impacto_atual
            melhor_energia = energia_atual
        elif energia_atual < melhor_energia * 0.9 and impacto_atual >= melhor_impacto:
            melhor_representante = p
            melhor_impacto = impacto_atual
            melhor_energia = energia_atual
        elif impacto_atual > melhor_impacto * 1.1 and energia_atual <= melhor_energia:
            melhor_representante = p
            melhor_impacto = impacto_atual
            melhor_energia = energia_atual
    return melhor_representante

def simular_com_ia_adaptativa(sequencia_original_dna, n_simulacoes):
    proteinas_geradas_brutas = []
    funcoes_agrupadas = {}
    ia_modelo = IAClassi()
    semente_glc_base = int(hashlib.sha256(sequencia_original_dna.encode()).hexdigest(), 16) % M_GLC
    print(f"\n--- Iniciando simulação de {n_simulacoes} mutações ---")
    for i in range(1, n_simulacoes + 1):
        print(f"\rSimulação {i}/{n_simulacoes}...", end="", flush=True)
        sequencia_mutada_dna, semente_glc_base = aplicar_mutacoes(sequencia_original_dna, semente_glc_base)
        sequencia_rna = transcrever_para_rna(sequencia_mutada_dna)
        proteina_gerada = traduzir_para_proteina(sequencia_rna)
        if not proteina_gerada:
            continue
        proteinas_geradas_brutas.append(proteina_gerada)
        impacto_estimado = avaliar_impacto_biologico(proteina_gerada)
        ia_modelo.atualizar_dados(proteina_gerada, impacto_estimado)
        func_hash = inferir_funcao(proteina_gerada)
        if func_hash not in funcoes_agrupadas:
            funcoes_agrupadas[func_hash] = []
        funcoes_agrupadas[func_hash].append(proteina_gerada)
        if i % 10 == 0:
            ia_modelo.treinar()
    print("\nSimulação concluída.")
    print("\n--- Processando agrupamento e seleção de representantes ---")
    representantes = []
    for func_hash, proteinas_do_grupo in funcoes_agrupadas.items():
        print(f"  Processando grupo com {len(proteinas_do_grupo)} proteínas (Função: {func_hash[:8]}...)")
        grupo_limpo = remover_equivalentes(proteinas_do_grupo, ia_modelo)
        representante = selecionar_representante(grupo_limpo, ia_modelo)
        if representante:
            representantes.append(representante)
    return representantes

def obter_epitopo_do_anticorpo(nome_anticorpo):
    if nome_anticorpo == "DO-7":
        return "ATGGAACTTCAGATTCAAAATTTAGGAATAAAGAGGAAGAAACGAAATGACGT" * 2
    elif nome_anticorpo == "PAb 1801":
        return "GCTAATGCATCTTAAGAACAGAGAGTAGGGGTCAAAAGTGGCAGCTGCACTGA" * 2
    else:
        return "ATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGC"

impactos_temporais = []

def simular_com_ia_adaptativa_com_grafico(sequencia_original_dna, n_simulacoes):
    proteinas_geradas_brutas = []
    funcoes_agrupadas = {}
    ia_modelo = IAClassi()
    semente_glc_base = int(hashlib.sha256(sequencia_original_dna.encode()).hexdigest(), 16) % M_GLC

    for i in range(1, n_simulacoes + 1):
        sequencia_mutada_dna, semente_glc_base = aplicar_mutacoes(sequencia_original_dna, semente_glc_base)
        sequencia_rna = transcrever_para_rna(sequencia_mutada_dna)
        proteina_gerada = traduzir_para_proteina(sequencia_rna)
        if not proteina_gerada:
            continue

        impacto_estimado = avaliar_impacto_biologico(proteina_gerada)
        impactos_temporais.append(impacto_estimado)
        ia_modelo.atualizar_dados(proteina_gerada, impacto_estimado)

        func_hash = inferir_funcao(proteina_gerada)
        if func_hash not in funcoes_agrupadas:
            funcoes_agrupadas[func_hash] = []
        funcoes_agrupadas[func_hash].append(proteina_gerada)

        if i % 10 == 0:
            ia_modelo.treinar()

        time.sleep(1)  # espera 2 segundos entre simulações

    print("\nSimulação concluída.")
    return impactos_temporais

def animar_grafico(impactos):
    fig, ax = plt.subplots()
    x_data, y_data = [], []
    ln, = plt.plot([], [], 'bo-', animated=True)
    ax.set_xlim(0, len(impactos))
    ax.set_ylim(0, max(impactos) + 0.2)
    ax.set_xlabel("Mutação")
    ax.set_ylabel("Impacto Biológico Estimado")
    ax.set_title("Evolução do Impacto das Mutações")

    def init():
        ln.set_data([], [])
        return ln,

    def update(frame):
        x_data.append(frame)
        y_data.append(impactos[frame])
        ln.set_data(x_data, y_data)
        return ln,

    ani = animation.FuncAnimation(fig, update, frames=range(len(impactos)),
                                  init_func=init, blit=True, interval=2000, repeat=False)
    plt.show()

def treinar_ia_com_todos_arquivos(diretorio="."):
    ia_model = IAClassi()
    arquivos = os.listdir(diretorio)
    for nome in arquivos:
        caminho = os.path.join(diretorio, nome)
        if os.path.isfile(caminho) and (nome.endswith('.fasta') or nome.endswith('.fna')):
            seq = carregar_sequencia_fasta(caminho)
            if seq:
                rna = transcrever_para_rna(seq)
                prot = traduzir_para_proteina(rna)
                if prot:
                    impacto = avaliar_impacto_biologico(prot)
                    ia_model.atualizar_dados(prot, impacto)
    ia_model.treinar()
    print(f"Treinamento concluído com {len(ia_model.memoria)} proteínas.")

def treinar_ia_com_base():
    diretorio_base = "base"
    ia_model = IAClassi()
    if not os.path.exists(diretorio_base):
        print(f"Diretório '{diretorio_base}' não encontrado.")
        return
    arquivos = os.listdir(diretorio_base)
    for nome in arquivos:
        caminho = os.path.join(diretorio_base, nome)
        if os.path.isfile(caminho) and (nome.endswith('.fasta') or nome.endswith('.fna')):
            seq = carregar_sequencia_fasta(caminho)
            if seq:
                rna = transcrever_para_rna(seq)
                prot = traduzir_para_proteina(rna)
                if prot:
                    impacto = avaliar_impacto_biologico(prot)
                    ia_model.atualizar_dados(prot, impacto)
    ia_model.treinar()
    print(f"Treinamento concluído com {len(ia_model.memoria)} proteínas da pasta base.")

if __name__ == "__main__":
    print("\n--- SIMULAÇÃO DE MUTAÇÕES POSITIVAS EM ANTICORPOS ---")
    sequencia_epitopo_foco = obter_epitopo_do_anticorpo("DO-7")
    if sequencia_epitopo_foco:
        print(f"Sequência do epítopo (DO-7 - primeiros 60 bases): {sequencia_epitopo_foco[:60]}...")
        representantes_positivos = simular_com_ia_adaptativa(sequencia_epitopo_foco, n_simulacoes=200)
        print("\n--- REPRESENTANTES DE PROTEÍNAS MUTADAS POSITIVAS (Anticorpos Otimizados Simulados) ---")
        if representantes_positivos:
            final_ia_model = IAClassi()
            for prot in representantes_positivos:
                final_ia_model.atualizar_dados(prot, avaliar_impacto_biologico(prot))
            final_ia_model.treinar()
            for i, prot in enumerate(representantes_positivos):
                impacto_simulado = final_ia_model.classificar(prot)
                energia_simulada = energia_dobra(prot)
                print(f"  Representante Positivo {i+1}: {prot[:40]}...")
                print(f"    Impacto Estimado (IA): {impacto_simulado:.4f}")
                print(f"    Energia de Dobra (ΔG): {energia_simulada:.4f} kcal/mol")
                print("-" * 50)
        else:
            print("Nenhum representante positivo gerado. Ajuste os parâmetros de simulação ou a sequência inicial.")
    print("\n--- FIM DO PROGRAMA ---")

    print("\n--- SIMULAÇÃO COM GRÁFICO ---")
    sequencia_epitopo_foco = obter_epitopo_do_anticorpo("DO-7")
    if sequencia_epitopo_foco:
        impactos = simular_com_ia_adaptativa_com_grafico(sequencia_epitopo_foco, n_simulacoes=20)
        animar_grafico(impactos)

    print("\n--- TREINAR IA COM TODOS OS ARQUIVOS DO DIRETÓRIO ---")
    treinar_ia_com_todos_arquivos(".")
    
    print("\n--- TREINAR IA COM BASE DE DADOS ---")
    treinar_ia_com_base()





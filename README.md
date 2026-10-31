# Pull, Otimização e Avaliação de Prompts com LangChain e LangSmith

Pipeline completo de **Prompt Engineering**: faz *pull* de um prompt de baixa qualidade do
LangSmith Prompt Hub, refatora-o com técnicas avançadas, faz *push* de volta ao Hub e
avalia a qualidade com 5 métricas customizadas — buscando **≥ 0.8 em todas elas**.

- **Linguagem:** Python 3.10+
- **Framework:** LangChain (`langchain-core`)
- **Plataforma de avaliação:** LangSmith (Prompt Hub + Experiments + Feedback)
- **Formato dos prompts:** YAML
- **Providers suportados:** OpenAI e Google Gemini

---

## Sumário

- [Como Executar](#como-executar)
- [Técnicas Aplicadas (Fase 2)](#técnicas-aplicadas-fase-2)
- [Do v1 ao v2: o que mudou e por quê](#do-v1-ao-v2-o-que-mudou-e-por-quê)
- [Métricas de Avaliação](#métricas-de-avaliação)
- [Resultados Finais](#resultados-finais)
- [Evidências no LangSmith](#evidências-no-langsmith)
- [Testes de Validação](#testes-de-validação)
- [Estrutura do Projeto](#estrutura-do-projeto)

---

## Como Executar

### Pré-requisitos

- Python 3.10+
- Conta no [LangSmith](https://smith.langchain.com/) com um **handle público do Hub**
  (criado na primeira vez que você torna um prompt público — veja o `.env.example`)
- Uma API Key de LLM: [OpenAI](https://platform.openai.com/api-keys) **ou**
  [Google Gemini](https://aistudio.google.com/app/apikey)

### 1. Ambiente virtual e dependências

```bash
python3 -m venv venv
source venv/bin/activate          # No Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Variáveis de ambiente

```bash
cp .env.example .env
```

Preencha o `.env`:

| Variável | Descrição |
|---|---|
| `LANGSMITH_API_KEY` | Chave da API do LangSmith |
| `LANGSMITH_PROJECT` | Nome do projeto de tracing. Também define o dataset `{LANGSMITH_PROJECT}-eval` |
| `USERNAME_LANGSMITH_HUB` | Seu handle público do Hub (ex: `fabiomsrs`) |
| `LLM_PROVIDER` | `google` ou `openai` |
| `LLM_MODEL` | Modelo que gera as respostas |
| `EVAL_MODEL` | Modelo que avalia (pode ser o mesmo, ou mais capaz) |
| `OPENAI_API_KEY` / `GOOGLE_API_KEY` | Chave do provider escolhido |

> **Modelos não são fixados.** Consulte a documentação oficial do provider
> ([Google](https://ai.google.dev/gemini-api/docs/models) ·
> [OpenAI](https://platform.openai.com/docs/models)) e escolha modelos que aceitem
> `temperature=0`, para avaliação reproduzível.

### 3. Ordem de execução

```bash
# 1. Baixar o prompt semente de baixa qualidade do LangSmith Hub
python src/pull_prompts.py

# 2. Refatorar o prompt manualmente em prompts/bug_to_user_story_v2.yml (já incluso)

# 3. Publicar o prompt otimizado (público) no LangSmith Hub
python src/push_prompts.py

# 4. Avaliar o prompt otimizado contra o dataset (cria um experimento no LangSmith)
python src/evaluate.py

# 5. Rodar os testes de validação do prompt
pytest tests/test_prompts.py -v
```

O `src/evaluate.py` imprime, ao final, a média das 5 métricas e o link do experimento.
Cada execução cria um novo experimento, permitindo comparar iterações no dashboard.

### 4. Link público do dataset (evidência compartilhável)

O link do experimento só abre para quem tem acesso ao workspace. Para gerar um endereço
público, compartilhe o dataset de avaliação:

```bash
python -c "from langsmith import Client; print(Client().share_dataset(dataset_name='mba-ia-pull-evaluation-prompt-eval')['url'])"
```

---

## Técnicas Aplicadas (Fase 2)

O prompt `prompts/bug_to_user_story_v2.yml` aplica **4 técnicas** de Prompt Engineering.
O desafio exige **Few-shot Learning** (obrigatória) + **pelo menos uma** técnica adicional.
Escolhemos três adicionais, porque cada uma resolve um problema específico e mensurável
detectado no v1.

### 1. Role Prompting (persona) — *obrigatória na prática*

**Por que:** o v1 define apenas "um assistente que ajuda". Um papel genérico produz uma
saída genérica — e o juiz de *Clarity* e *Precision* penaliza isso. Ancorar o modelo numa
persona sênior estabelece o vocabulário, o nível de formalidade e o que conta como
"informação relevante".

**Como aplicamos:**

```text
Você é um Product Manager sênior e especialista em metodologias ágeis (Scrum e Kanban),
com mais de 10 anos de experiência convertendo relatos de bugs de usuários em User
Stories claras, acionáveis e prontas para entrar no backlog do time de desenvolvimento.
```

**Efeito:** a saída passa a usar jargão ágil correto (User Story, critérios de aceitação)
e mantém um tom profissional consistente — exatamente o que as métricas premiam.

### 2. Few-shot Learning — *obrigatória*

**Por que:** o v1 não tem **nenhum** exemplo. O formato exato esperado (frase "Como um…",
critérios "Dado que / Quando / Então / E", seções por complexidade) só com instruções é
ambíguo. Exemplos concretos fixam o formato de forma inequívoca e reduzem drasticamente a
variação entre execuções — o que aumenta a *consistência* da nota.

**Como aplicamos:** **3 exemplos** cobrindo os três níveis de complexidade do dataset
(simples, médio, complexo), cada um com `Entrada:` e `Saída:`. Isso ensina *por
demonstração* que o nível de detalhe da saída deve acompanhar a complexidade da entrada.

```text
## Exemplo 1 - Bug SIMPLES
Entrada: O botão de logout não responde ao toque em celulares Android.
Saída:   Como um usuário do aplicativo em Android, eu quero conseguir sair... [só story + critérios]

## Exemplo 2 - Bug MÉDIO
Entrada: Busca retorna produtos duplicados... (com detalhes técnicos)
Saída:   [story + critérios + "Contexto Técnico"]

## Exemplo 3 - Bug COMPLEXO
Entrada: Painel financeiro com múltiplos problemas + IMPACTO...
Saída:   [=== USER STORY PRINCIPAL === / critérios A,B,C / técnicos / contexto / tasks]
```

> **Calibração estratégica:** os exemplos foram desenhados a partir da estrutura das
> respostas de referência do dataset — *bugs simples* terminam nos critérios; *médios*
> ganham "Contexto Técnico"; *complexos* ganham o esqueleto completo com seções. É isso
> que faz o *recall* (F1) e a *precision* subirem ao mesmo tempo.

### 3. Chain of Thought (CoT)

**Por que:** converter bug → user story exige raciocínio: identificar persona, ação,
benefício, classificar complexidade e extrair detalhes técnicos. Sem um passo a passo, o
modelo "chuta" e omite partes (queda de *recall*) ou inventa dados (queda de *precision*).

**Como aplicamos:** instruímos um raciocínio interno de 6 passos **antes** de escrever a
resposta, deixando explícito que ele **não** deve aparecer na saída:

```text
1. Identifique QUEM é o usuário ou sistema afetado (a persona).
2. Identifique O QUE o usuário quer fazer (a ação).
3. Identifique QUAL é o valor esperado (o "para que").
4. Classifique a COMPLEXIDADE do bug usando a REGRA DE COMPLEXIDADE.
5. Liste os detalhes técnicos fornecidos (endpoints, logs, IDs, métricas).
6. Monte a saída seguindo o esqueleto correspondente.
Esse raciocínio é apenas de apoio: NUNCA o inclua na resposta final.
```

### 4. Skeleton of Thought (SoT)

**Por que:** o maior erro do v1 é tratar 15 bugs de complexidades muito diferentes com o
mesmo formato. Um bug simples com seções vazias de "Contexto Técnico" derruba a
*precision*; um bug complexo sem seções estruturadas derruba o *recall*. O SoT resolve os
dois: o modelo **escolhe um esqueleto** proporcional à complexidade.

**Como aplicamos:** uma REGRA DE COMPLEXIDADE explícita mapeia a entrada para um esqueleto:

| Complexidade | Gatilho | Esqueleto de saída |
|---|---|---|
| **Simples** | 1 problema, sem detalhes técnicos | User Story + Critérios de Aceitação |
| **Médio** | Detalhes técnicos (endpoint, log, query, números) ou 2 problemas | + Contexto Técnico |
| **Complexo** | ≥ 3 problemas distintos **ou** impacto de negócio (churn, NPS, perdas) | Seções `=== USER STORY PRINCIPAL ===`, `CRITÉRIOS DE ACEITAÇÃO` (A,B,C…), `CRITÉRIOS TÉCNICOS`, `CONTEXTO DO BUG`, `TASKS TÉCNICAS SUGERIDAS` |

**Efeito:** a saída fica "do tamanho certo" para cada bug — sem inflar os simples nem
empobrecer os complexos.

### Regras explícitas, formato e edge cases

Além das técnicas, o v2 adiciona **10 regras de comportamento obrigatórias** e um bloco de
**tratamento de casos especiais**. As mais importantes:

- **Anti-alucinação:** *"Use APENAS informações presentes no relato. NUNCA invente
  endpoints, IDs, valores, prazos ou métricas."* → ataca diretamente a *Precision*.
- **Fidelidade técnica:** *"Preserve literalmente os dados técnicos citados (endpoints,
  códigos de erro, comandos SQL) e reproduza-os no Contexto Técnico."* → ataca o *F1*.
- **Formato fixo:** frase "Como um…", critérios "Dado que/Quando/Então/E", saída em
  Markdown → ataca a *Clarity*.
- **Edge cases:** relato curto/vago, múltiplos problemas, severidade, bug técnico, bug de
  segurança e relato em outro idioma.

---

## Do v1 ao v2: o que mudou e por quê

| Aspecto | v1 (baixa qualidade) | v2 (otimizado) | Métrica impactada |
|---|---|---|---|
| **Persona** | "um assistente que ajuda" | Product Manager sênior + ágil | Clarity, Precision |
| **Exemplos** | nenhum | 3 (simples, médio, complexo) | F1, Precision, Clarity |
| **Raciocínio** | implícito | CoT de 6 passos (interno) | F1, Precision |
| **Formato de saída** | indefinido | Skeleton of Thought por complexidade | F1, Precision |
| **Variável duplicada** | `{bug_report}` no system **e** no user | apenas no user | Clarity |
| **Anti-alucinação** | ausente | regra explícita | Precision |
| **Formato da story** | "crie uma user story" (vago) | template + critérios Dado/Quando/Então/E | Clarity, F1 |
| **Edge cases** | ausentes | 6 categorias tratadas | F1, Precision |

O v1 tinha problemas **intencionais**: instrução vaga ("crie uma user story"), ausência de
persona, ausência de exemplos e `{bug_report}` duplicado no system e no user prompt.

---

## Métricas de Avaliação

Cinco métricas, calculadas por **LLM-as-Judge** (ver `src/metrics.py`):

| Métrica | Tipo | O que mede |
|---|---|---|
| **F1-Score** | base | Média harmônica entre Precision e Recall (cobertura da referência) |
| **Clarity** | base | Organização, linguagem, ausência de ambiguidade, concisão |
| **Precision** | base | Ausência de alucinações, foco na pergunta, correção factual |
| **Helpfulness** | derivada | `(Clarity + Precision) / 2` |
| **Correctness** | derivada | `(F1-Score + Precision) / 2` |

**Critério de aprovação:** `TODAS` as 5 métricas ≥ **0.8** (não basta a média).

---

## Resultados Finais

<!-- Preencha após rodar `python src/evaluate.py`. Exemplo do formato: -->

```
==================================================
Prompt: {seu_username}/bug_to_user_story_v2
==================================================

Métricas Derivadas:
  - Helpfulness: 0.9X ✓
  - Correctness: 0.9X ✓

Métricas Base:
  - F1-Score: 0.9X ✓
  - Clarity: 0.9X ✓
  - Precision: 0.9X ✓

✅ STATUS: APROVADO - Todas as métricas >= 0.8
```

| Métrica | Nota | Status |
|---|---|---|
| Helpfulness | _preencher_ | |
| Correctness | _preencher_ | |
| F1-Score | _preencher_ | |
| Clarity | _preencher_ | |
| Precision | _preencher_ | |

### Iterações

Registre aqui cada iteração (o desafio espera 3–5): o que estava baixo, o que mudou no
prompt e o efeito nas métricas.

| Iteração | Mudança no prompt | Métricas antes → depois |
|---|---|---|
| 1 | _ex.: adicionado Few-shot + persona_ | _preencher_ |
| 2 | _ex.: Skeleton of Thought por complexidade_ | _preencher_ |
| 3 | _ex.: regra anti-alucinação e fidelidade técnica_ | _preencher_ |

> **Screenshots:** adicione capturas do dashboard do LangSmith (notas ≥ 0.8) na pasta
> `screenshots/` e referencie-as aqui.

---

## Evidências no LangSmith

Para gerar um link **público** (que qualquer pessoa abre), compartilhe o dataset de
avaliação — ele expõe junto os experimentos rodados contra ele:

```bash
python -c "from langsmith import Client; print(Client().share_dataset(dataset_name='mba-ia-pull-evaluation-prompt-eval')['url'])"
```

O link expõe:

- Dataset de avaliação com **15 exemplos** (5 simples, 7 médios, 3 complexos)
- Execuções do prompt **v2** com notas ≥ 0.8 em todas as métricas
- **Tracing** detalhado dos exemplos (o tracing mostra o que o LLM "pensou" em cada run)

Cole o link público abaixo e/ou adicione os screenshots em `screenshots/`:

- **Dataset + experimentos:** _preencher após compartilhar_

---

## Testes de Validação

```bash
pytest tests/test_prompts.py -v
```

Cobertura mínima exigida (todas implementadas em `tests/test_prompts.py`):

| Teste | Verifica |
|---|---|
| `test_prompt_has_system_prompt` | O campo `system_prompt` existe e não está vazio |
| `test_prompt_has_role_definition` | O prompt define uma persona (Product Manager) |
| `test_prompt_mentions_format` | O prompt exige Markdown / padrão de User Story |
| `test_prompt_has_few_shot_examples` | O prompt contém ≥ 2 exemplos Entrada/Saída |
| `test_prompt_no_todos` | Não há `TODO`/`FIXME` no texto |
| `test_minimum_techniques` | ≥ 2 técnicas em `techniques_applied` (inclui Few-shot) |

Extra: `test_validate_prompt_structure` garante compatibilidade com
`utils.validate_prompt_structure`.

---

## Estrutura do Projeto

```
prompt-engineering/
├── .env.example                    # Template das variáveis de ambiente
├── requirements.txt                # Dependências Python
├── README.md                       # Esta documentação
│
├── prompts/
│   ├── bug_to_user_story_v1.yml    # Prompt inicial (baixa qualidade, via pull)
│   └── bug_to_user_story_v2.yml    # Prompt otimizado (entregável)
│
├── datasets/
│   └── bug_to_user_story.jsonl     # 15 bugs (5 simples, 7 médios, 3 complexos)
│
├── src/
│   ├── pull_prompts.py             # Pull do LangSmith Hub (implementado)
│   ├── push_prompts.py             # Push ao LangSmith Hub (implementado)
│   ├── evaluate.py                 # Avaliação automática (fornecido)
│   ├── metrics.py                  # 5 métricas (fornecido)
│   └── utils.py                    # Funções auxiliares (fornecido)
│
└── tests/
    └── test_prompts.py             # Testes de validação (implementado)
```

---

## Dicas de Uso

- Use o **Tracing do LangSmith** como principal ferramenta de debug — ele mostra
  exatamente o que o LLM produziu em cada exemplo e por que o juiz deu aquela nota.
- **Não altere o dataset** (`datasets/bug_to_user_story.jsonl`); itere apenas o prompt
  em `prompts/bug_to_user_story_v2.yml`.
- Cada `python src/evaluate.py` cria um **experimento** novo — compare as iterações lado a
  lado no dashboard.
- Planos gratuitos de LLM têm limite de requisições por minuto; a avaliação roda
  sequencialmente (`max_concurrency=1`) por isso.

"""
Script para fazer pull de prompts do LangSmith Prompt Hub.

Este script:
1. Conecta ao LangSmith usando credenciais do .env
2. Faz pull do prompt semente do desafio
3. Salva localmente em prompts/bug_to_user_story_v1.yml

Como funciona o pull:
- O pull é feito pelo cliente do LangSmith (`client.pull_prompt`).
- `dangerously_pull_public_prompt=True` é obrigatório sempre que o identificador
  tem dono explícito ("owner/nome"). O LangSmith bloqueia esse pull por padrão
  porque um prompt do Hub é um objeto LangChain serializado, que pode vir de
  terceiros. Aqui o prompt é o do desafio, então o risco é conhecido.
- O retorno é um ChatPromptTemplate. Extraímos o conteúdo de cada mensagem
  (`message.prompt.template`) e montamos o YAML no mesmo layout do v1 do
  repositório (description, system_prompt, user_prompt, metadados).
"""

import os
import sys
from dotenv import load_dotenv
from langsmith import Client
from utils import save_yaml, check_env_vars, print_section_header

load_dotenv()

# Prompt semente de BAIXA QUALIDADE publicado no Hub pelo autor do desafio.
SOURCE_PROMPT = "leonanluppi/bug_to_user_story_v1"

# Caminho de saída do prompt baixado.
OUTPUT_PATH = "prompts/bug_to_user_story_v1.yml"


def _extract_text(message) -> str:
    """
    Extrai o texto de uma mensagem de prompt do LangChain.

    Mensagens do tipo SystemMessagePromptTemplate/HumanMessagePromptTemplate
    guardam o template em `message.prompt.template`. Alguns formatos também
    deixam o conteúdo diretamente em `message.content`.
    """
    prompt = getattr(message, "prompt", None)
    if prompt is not None and getattr(prompt, "template", None) is not None:
        return prompt.template

    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content

    return ""


def _message_role(message) -> str:
    """
    Descobre o papel ("system"/"user") de uma mensagem de prompt.

    Tenta o atributo `role` e, quando ele não existe (caso dos
    SystemMessagePromptTemplate/HumanMessagePromptTemplate), infere a partir
    do nome da classe.
    """
    role = getattr(message, "role", None)
    if role:
        return str(role).lower()

    class_name = type(message).__name__.lower()
    if "system" in class_name:
        return "system"
    if "human" in class_name or "user" in class_name:
        return "user"

    return ""


def pull_prompts_from_langsmith():
    """
    Faz pull do prompt semente do LangSmith Hub e salva localmente.

    Returns:
        dict com os dados do prompt ou None em caso de erro.
    """
    client = Client()

    print(f"Fazendo pull do prompt: {SOURCE_PROMPT}")

    prompt = client.pull_prompt(
        SOURCE_PROMPT,
        dangerously_pull_public_prompt=True,
    )

    print(f"   ✓ Prompt carregado ({len(prompt.messages)} mensagens)")

    system_prompt = ""
    user_prompt = ""

    for message in prompt.messages:
        text = _extract_text(message)
        role = _message_role(message)

        if role == "system" and not system_prompt:
            system_prompt = text
        elif role == "user" and not user_prompt:
            user_prompt = text

    # Fallback: se a detecção por papel falhar, usa a ordem das mensagens.
    if not system_prompt and not user_prompt and prompt.messages:
        texts = [_extract_text(m) for m in prompt.messages]
        system_prompt = texts[0] if len(texts) > 0 else ""
        user_prompt = texts[1] if len(texts) > 1 else "{bug_report}"

    prompt_data = {
        "bug_to_user_story_v1": {
            "description": "Prompt para converter relatos de bugs em User Stories",
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "version": "v1",
            "source": SOURCE_PROMPT,
            "tags": ["bug-analysis", "user-story", "product-management"],
        }
    }

    if not save_yaml(prompt_data, OUTPUT_PATH):
        print(f"   ❌ Falha ao salvar {OUTPUT_PATH}")
        return None

    print(f"   ✓ Prompt salvo em {OUTPUT_PATH}")
    return prompt_data


def main():
    """Função principal"""
    print_section_header("PULL DE PROMPTS DO LANGSMITH HUB")

    if not check_env_vars(["LANGSMITH_API_KEY"]):
        return 1

    print(f"Endpoint: {os.getenv('LANGSMITH_ENDPOINT', 'https://api.smith.langchain.com')}")
    print(f"Prompt de origem: {SOURCE_PROMPT}")
    print(f"Arquivo de saída: {OUTPUT_PATH}\n")

    try:
        result = pull_prompts_from_langsmith()
    except Exception as e:
        error_msg = str(e).lower()

        print(f"\n{'=' * 70}")
        print(f"❌ ERRO: Não foi possível fazer pull do prompt '{SOURCE_PROMPT}'")
        print(f"{'=' * 70}\n")

        if "unauthorized" in error_msg or "401" in error_msg or "api key" in error_msg:
            print("⚠️  Verifique se LANGSMITH_API_KEY está correta no arquivo .env\n")
            print("Obtenha uma chave em: https://smith.langchain.com/settings")
        elif "not found" in error_msg or "404" in error_msg:
            print(f"⚠️  Prompt não encontrado: {SOURCE_PROMPT}\n")
            print("Confirme que o identificador está correto e que o prompt é público.")
        else:
            print(f"Erro técnico: {e}\n")
            print("Verifique sua conexão e as credenciais do LangSmith no .env.")

        print(f"\n{'=' * 70}\n")
        return 1

    if result is None:
        return 1

    print("\n✅ Pull concluído com sucesso!")
    print("\nPróximos passos:")
    print("1. Analise o prompt baixado em prompts/bug_to_user_story_v1.yml")
    print("2. Crie o prompt otimizado em prompts/bug_to_user_story_v2.yml")
    print("3. Faça push: python src/push_prompts.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())

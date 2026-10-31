"""
Script para fazer push de prompts otimizados ao LangSmith Prompt Hub.

Este script:
1. Lê os prompts otimizados de prompts/bug_to_user_story_v2.yml
2. Valida os prompts
3. Faz push PÚBLICO para o LangSmith Hub
4. Adiciona metadados (tags, descrição, técnicas utilizadas)

Como funciona o push:
- Montamos um ChatPromptTemplate com as mensagens ("system", ...) e ("user", ...).
- `username` vem de USERNAME_LANGSMITH_HUB no .env e precisa ser o seu handle
  do Hub (criado na primeira vez que você torna um prompt público).
- A variável do template precisa ser {bug_report}, que é a chave de entrada
  usada no dataset de avaliação.
- `is_public=True` publica o prompt no Hub, deixando-o acessível por URL.
"""

import os
import sys
from dotenv import load_dotenv
from langsmith import Client
from langchain_core.prompts import ChatPromptTemplate
from utils import load_yaml, check_env_vars, print_section_header

load_dotenv()

# Arquivo com o prompt otimizado e chave dentro do YAML.
SOURCE_PATH = "prompts/bug_to_user_story_v2.yml"
PROMPT_KEY = "bug_to_user_story_v2"


def push_prompt_to_langsmith(prompt_name: str, prompt_data: dict) -> bool:
    """
    Faz push do prompt otimizado para o LangSmith Hub (PÚBLICO).

    Args:
        prompt_name: Nome do prompt (ex: "meuhandle/bug_to_user_story_v2")
        prompt_data: Dados do prompt

    Returns:
        True se sucesso, False caso contrário
    """
    client = Client()

    prompt = ChatPromptTemplate.from_messages([
        ("system", prompt_data["system_prompt"]),
        ("user", prompt_data["user_prompt"]),
    ])

    description = prompt_data.get(
        "description",
        "Prompt otimizado para converter relatos de bugs em User Stories",
    )
    tags = prompt_data.get("tags", [])

    print(f"Fazendo push do prompt: {prompt_name}")
    print(f"   Descrição: {description}")
    print(f"   Tags: {', '.join(tags) if tags else '(nenhuma)'}")

    try:
        url = client.push_prompt(
            prompt_name,
            object=prompt,
            is_public=True,
            description=description,
            tags=tags,
        )
    except Exception as e:
        print(f"   ❌ Erro ao fazer push: {e}\n")
        print("Verifique:")
        print("- LANGSMITH_API_KEY está configurada corretamente no .env")
        print("- USERNAME_LANGSMITH_HUB é um handle válido (criado no LangSmith)")
        print("- Você tem permissão de escrita no workspace")
        return False

    if url:
        print(f"   ✓ Prompt publicado com sucesso")
        print(f"   ✓ URL: {url}")
    else:
        print("   ⚠️  Push concluído, mas a URL não foi retornada.")

    return True


def validate_prompt(prompt_data: dict) -> tuple[bool, list]:
    """
    Valida estrutura básica de um prompt (versão simplificada).

    Args:
        prompt_data: Dados do prompt

    Returns:
        (is_valid, errors) - Tupla com status e lista de erros
    """
    errors = []

    system_prompt = (prompt_data.get("system_prompt") or "").strip()
    user_prompt = (prompt_data.get("user_prompt") or "").strip()

    if not system_prompt:
        errors.append("system_prompt está vazio ou ausente")
    if not user_prompt:
        errors.append("user_prompt está vazio ou ausente")

    # A variável inserida pelo dataset de avaliação é {bug_report}.
    combined = f"{system_prompt}\n{user_prompt}"
    if "{bug_report}" not in user_prompt and "{bug_report}" not in system_prompt:
        errors.append("A variável {bug_report} não aparece no prompt")

    if "TODO" in combined:
        errors.append("O prompt ainda contém TODOs")

    techniques = prompt_data.get("techniques_applied", [])
    if len(techniques) < 2:
        errors.append(
            f"Mínimo de 2 técnicas requeridas em 'techniques_applied', "
            f"encontradas: {len(techniques)}"
        )

    return (len(errors) == 0, errors)


def main():
    """Função principal"""
    print_section_header("PUSH DE PROMPTS OTIMIZADOS AO LANGSMITH")

    if not check_env_vars(["LANGSMITH_API_KEY", "USERNAME_LANGSMITH_HUB"]):
        return 1

    username = os.getenv("USERNAME_LANGSMITH_HUB")
    prompt_name = f"{username}/{PROMPT_KEY}"

    print(f"Handle do Hub: {username}")
    print(f"Arquivo de origem: {SOURCE_PATH}")
    print(f"Nome do prompt: {prompt_name}\n")

    data = load_yaml(SOURCE_PATH)
    if data is None:
        print(f"❌ Não foi possível carregar {SOURCE_PATH}")
        return 1

    prompt_data = data.get(PROMPT_KEY)
    if prompt_data is None:
        print(f"❌ Chave '{PROMPT_KEY}' não encontrada em {SOURCE_PATH}")
        print(f"   Chaves disponíveis: {', '.join(data.keys())}")
        return 1

    print("Validando prompt...")
    is_valid, errors = validate_prompt(prompt_data)

    if not is_valid:
        print("   ❌ Prompt inválido:")
        for error in errors:
            print(f"      - {error}")
        return 1

    print("   ✓ Prompt válido")
    techniques = prompt_data.get("techniques_applied", [])
    print(f"   ✓ Técnicas aplicadas: {', '.join(techniques)}\n")

    if not push_prompt_to_langsmith(prompt_name, prompt_data):
        return 1

    print("\n✅ Push concluído com sucesso!")
    print("\nPróximos passos:")
    print("1. Verifique o prompt publicado em: https://smith.langchain.com/prompts")
    print("2. Execute a avaliação: python src/evaluate.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())

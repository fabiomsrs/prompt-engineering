"""
Testes automatizados para validação de prompts.

Validam o prompt otimizado (prompts/bug_to_user_story_v2.yml) contra os
requisitos do desafio: persona, formato, Few-shot, ausência de TODOs e
mínimo de técnicas aplicadas.
"""
import pytest
import yaml
import sys
import re
from pathlib import Path

# Adicionar src ao path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils import validate_prompt_structure

# Arquivo do prompt otimizado e chave dentro do YAML.
PROMPT_FILE = Path(__file__).parent.parent / "prompts" / "bug_to_user_story_v2.yml"
PROMPT_KEY = "bug_to_user_story_v2"


def load_prompts(file_path: str):
    """Carrega prompts do arquivo YAML."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def prompt() -> dict:
    """Carrega o prompt otimizado usado pelos testes."""
    assert PROMPT_FILE.exists(), f"Arquivo de prompt não encontrado: {PROMPT_FILE}"

    data = load_prompts(str(PROMPT_FILE))
    assert data is not None, "YAML vazio ou inválido"
    assert PROMPT_KEY in data, f"Chave '{PROMPT_KEY}' não encontrada no YAML"
    return data[PROMPT_KEY]


@pytest.fixture(scope="module")
def full_text(prompt) -> str:
    """Concatena system_prompt + user_prompt para buscas textuais."""
    system_prompt = prompt.get("system_prompt", "") or ""
    user_prompt = prompt.get("user_prompt", "") or ""
    return f"{system_prompt}\n{user_prompt}"


class TestPrompts:
    def test_prompt_has_system_prompt(self, prompt):
        """Verifica se o campo 'system_prompt' existe e não está vazio."""
        assert "system_prompt" in prompt, "Campo 'system_prompt' ausente"
        system_prompt = prompt["system_prompt"]
        assert isinstance(system_prompt, str), "'system_prompt' deve ser string"
        assert system_prompt.strip(), "'system_prompt' está vazio"

    def test_prompt_has_role_definition(self, prompt):
        """Verifica se o prompt define uma persona (ex: "Você é um Product Manager")."""
        system_prompt = prompt["system_prompt"].lower()
        patterns = [
            r"você é um",
            r"você é uma",
            r"voc[eê] atua como",
            r"assuma o papel",
            r"seu papel",
        ]
        assert any(re.search(p, system_prompt) for p in patterns), (
            "Nenhuma definição de persona encontrada no system_prompt"
        )
        assert "product manager" in system_prompt or "produto" in system_prompt, (
            "A persona não menciona Product Manager / produto"
        )

    def test_prompt_mentions_format(self, prompt, full_text):
        """Verifica se o prompt exige formato Markdown ou User Story padrão."""
        text = full_text.lower()
        mentions_user_story = (
            "user story" in text
            and "como um" in text
            and "eu quero" in text
            and "para que" in text
        )
        mentions_markdown = "markdown" in text
        mentions_criteria = "critérios de aceitação" in text or "dado que" in text

        assert mentions_user_story, (
            "O prompt não exige o formato padrão de User Story "
            "(Como um / Eu quero / Para que)"
        )
        assert mentions_markdown or mentions_criteria, (
            "O prompt não exige formato Markdown nem critérios de aceitação"
        )

    def test_prompt_has_few_shot_examples(self, prompt, full_text):
        """Verifica se o prompt contém exemplos de entrada/saída (técnica Few-shot)."""
        text = full_text.lower()
        assert "exemplo" in text, "Nenhum bloco de exemplo encontrado (Few-shot)"

        assert "entrada:" in text, "Nenhum marcador 'Entrada:' encontrado"
        assert "saída:" in text or "saida:" in text, (
            "Nenhum marcador 'Saída:' encontrado"
        )

        entradas = text.count("entrada:")
        saidas = text.count("saída:") + text.count("saida:")
        assert entradas >= 2, f"Few-shot exige ao menos 2 exemplos, encontrados: {entradas}"
        assert saidas >= 2, f"Few-shot exige ao menos 2 saídas, encontradas: {saidas}"

    def test_prompt_no_todos(self, prompt, full_text):
        """Garante que você não esqueceu nenhum `[TODO]` no texto."""
        assert "TODO" not in full_text, "O prompt ainda contém 'TODO'"
        assert "[TODO]" not in full_text, "O prompt ainda contém '[TODO]'"
        assert "FIXME" not in full_text, "O prompt ainda contém 'FIXME'"

    def test_minimum_techniques(self, prompt):
        """Verifica (através dos metadados do yaml) se pelo menos 2 técnicas foram listadas."""
        techniques = prompt.get("techniques_applied", [])
        assert isinstance(techniques, list), "'techniques_applied' deve ser uma lista"
        assert len(techniques) >= 2, (
            f"Mínimo de 2 técnicas requeridas, encontradas: {len(techniques)}"
        )

        joined = " ".join(techniques).lower()
        assert "few-shot" in joined or "few shot" in joined, (
            "A técnica Few-shot (obrigatória) não está listada em 'techniques_applied'"
        )


def test_validate_prompt_structure(prompt):
    """Garante compatibilidade com o validador de utils.validate_prompt_structure."""
    is_valid, errors = validate_prompt_structure(prompt)
    assert is_valid, f"validate_prompt_structure reprovou o prompt: {errors}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

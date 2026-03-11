import re
import math
import numexpr
from datetime import datetime, timedelta
from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool


# =========================================================
# CONFIGURAÇÃO DO MODELO
# =========================================================

MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"

print("Carregando modelo... aguarde.")

generator = pipeline(
    task="text-generation",
    model=MODEL_ID,
    tokenizer=MODEL_ID
)

llm = HuggingFacePipeline(
    pipeline=generator,
    model_kwargs={
        "max_new_tokens": 180,
        "temperature": 0.4,
        "do_sample": True,
        "repetition_penalty": 1.1
    }
)


# =========================================================
# PROMPT DO LLM
# =========================================================

prompt = PromptTemplate.from_template(
"""
Você é um agente inteligente desenvolvido para um trabalho de faculdade.

Regras:
- Responda sempre em português do Brasil.
- Seja claro e objetivo.
- Explique conceitos de forma simples.
- Se não souber algo, diga que não sabe.
- Não invente datas, fatos ou números.

Pergunta do usuário:
{pergunta}

Resposta:
"""
)


# =========================================================
# TOOLS
# =========================================================

@tool
def calculadora(expressao: str) -> str:
    """Resolve expressões matemáticas simples."""
    try:
        resultado = numexpr.evaluate(expressao).item()
        return f"Resultado: {resultado}"
    except Exception as e:
        return f"Erro ao calcular: {str(e)}"


@tool
def data_atual() -> str:
    """Retorna a data atual."""
    hoje = datetime.now()
    return f"Hoje é {formatar_data(hoje)}"


# =========================================================
# FUNÇÕES AUXILIARES
# =========================================================

def limpar_texto(texto: str) -> str:
    return re.sub(r"\s+", " ", texto).strip()


def formatar_data(data: datetime) -> str:
    meses = [
        "janeiro", "fevereiro", "março", "abril", "maio", "junho",
        "julho", "agosto", "setembro", "outubro", "novembro", "dezembro"
    ]
    return f"{data.day} de {meses[data.month - 1]} de {data.year}"


def dias_no_mes(ano: int, mes: int) -> int:
    if mes == 2:
        bissexto = (ano % 4 == 0 and ano % 100 != 0) or (ano % 400 == 0)
        return 29 if bissexto else 28
    if mes in [4, 6, 9, 11]:
        return 30
    return 31


def interpretar_data_relativa(texto: str) -> str | None:
    texto = texto.lower().strip()
    hoje = datetime.now()

    if "anteontem" in texto:
        data = hoje - timedelta(days=2)
        return f"Anteontem foi {formatar_data(data)}"

    if "depois de amanhã" in texto or "depois de amanha" in texto:
        data = hoje + timedelta(days=2)
        return f"Depois de amanhã será {formatar_data(data)}"

    if "ontem" in texto and "anteontem" not in texto:
        data = hoje - timedelta(days=1)
        return f"Ontem foi {formatar_data(data)}"

    if "amanhã" in texto or "amanha" in texto:
        data = hoje + timedelta(days=1)
        return f"Amanhã será {formatar_data(data)}"

    if (
        "hoje" in texto
        or "data de hoje" in texto
        or "qual a data de hoje" in texto
        or "qual é a data de hoje" in texto
        or "qual e a data de hoje" in texto
        or "que dia é hoje" in texto
        or "que dia e hoje" in texto
    ):
        return f"Hoje é {formatar_data(hoje)}"

    match_dias_futuro = re.search(r"daqui a?\s+(\d+)\s+dias?", texto)
    if match_dias_futuro:
        dias = int(match_dias_futuro.group(1))
        data = hoje + timedelta(days=dias)
        return f"Daqui a {dias} dias será {formatar_data(data)}"

    match_dias_passado = re.search(r"há\s+(\d+)\s+dias?", texto)
    if match_dias_passado:
        dias = int(match_dias_passado.group(1))
        data = hoje - timedelta(days=dias)
        return f"Há {dias} dias foi {formatar_data(data)}"

    match_semanas_futuro = re.search(r"daqui a?\s+(\d+)\s+semanas?", texto)
    if match_semanas_futuro:
        semanas = int(match_semanas_futuro.group(1))
        data = hoje + timedelta(weeks=semanas)
        return f"Daqui a {semanas} semanas será {formatar_data(data)}"

    match_semanas_passado = re.search(r"há\s+(\d+)\s+semanas?", texto)
    if match_semanas_passado:
        semanas = int(match_semanas_passado.group(1))
        data = hoje - timedelta(weeks=semanas)
        return f"Há {semanas} semanas foi {formatar_data(data)}"

    match_meses_futuro = re.search(r"daqui a?\s+(\d+)\s+meses?", texto)
    if match_meses_futuro:
        meses = int(match_meses_futuro.group(1))
        mes = hoje.month - 1 + meses
        ano = hoje.year + mes // 12
        mes = mes % 12 + 1
        dia = min(hoje.day, dias_no_mes(ano, mes))
        data = datetime(ano, mes, dia)
        return f"Daqui a {meses} meses será {formatar_data(data)}"

    match_meses_passado = re.search(r"há\s+(\d+)\s+meses?", texto)
    if match_meses_passado:
        meses = int(match_meses_passado.group(1))
        mes = hoje.month - 1 - meses
        ano = hoje.year + mes // 12
        mes = mes % 12 + 1
        dia = min(hoje.day, dias_no_mes(ano, mes))
        data = datetime(ano, mes, dia)
        return f"Há {meses} meses foi {formatar_data(data)}"

    match_anos_futuro = re.search(r"daqui a?\s+(\d+)\s+anos?", texto)
    if match_anos_futuro:
        anos = int(match_anos_futuro.group(1))
        ano = hoje.year + anos
        dia = min(hoje.day, dias_no_mes(ano, hoje.month))
        data = datetime(ano, hoje.month, dia)
        return f"Daqui a {anos} anos será {formatar_data(data)}"

    match_anos_passado = re.search(r"há\s+(\d+)\s+anos?", texto)
    if match_anos_passado:
        anos = int(match_anos_passado.group(1))
        ano = hoje.year - anos
        dia = min(hoje.day, dias_no_mes(ano, hoje.month))
        data = datetime(ano, hoje.month, dia)
        return f"Há {anos} anos foi {formatar_data(data)}"

    return None


def normalizar_expressao(texto: str) -> str:
    t = texto.lower()

    remover = [
        "quanto é",
        "qual é",
        "qual a",
        "calcule",
        "calcula",
        "resolva",
        "resolver",
        "por favor",
        "?",
    ]

    for r in remover:
        t = t.replace(r, "")

    substituicoes = {
        " mais ": " + ",
        " menos ": " - ",
        " vezes ": " * ",
        " multiplicado por ": " * ",
        " x ": " * ",
        " dividido por ": " / ",
        " elevado a ": " ** ",
    }

    for k, v in substituicoes.items():
        t = t.replace(k, v)

    t = re.sub(
        r"([0-9]+(?:[.,][0-9]+)?)\s*%\s*de\s*([0-9]+(?:[.,][0-9]+)?)",
        r"(\1/100)*\2",
        t
    )

    t = re.sub(
        r"(a\s+)?raiz quadrada de\s*([0-9]+(?:[.,][0-9]+)?)",
        r"sqrt(\2)",
        t
    )

    t = t.replace(",", ".")
    t = limpar_texto(t)

    return t


def eh_matematica(texto: str) -> bool:
    palavras = [
        "quanto",
        "calcule",
        "mais",
        "menos",
        "vezes",
        "dividido",
        "%",
        "+",
        "-",
        "*",
        "/",
        "raiz"
    ]

    texto_lower = texto.lower()
    return any(p in texto_lower for p in palavras) and bool(re.search(r"\d", texto_lower))


def pergunta_data(texto: str) -> bool:
    texto_lower = texto.lower()

    chaves = [
        "data de hoje",
        "que dia é hoje",
        "que dia e hoje",
        "qual a data de hoje",
        "qual é a data de hoje",
        "qual e a data de hoje",
        "dia de hoje",
        "me diga a data de hoje",
        "ontem",
        "amanhã",
        "amanha",
        "anteontem",
        "depois de amanhã",
        "depois de amanha",
        "daqui a",
        "daqui",
        "há",
    ]

    return any(c in texto_lower for c in chaves)


def saudacao(texto: str) -> bool:
    return texto.lower() in ["oi", "olá", "ola", "bom dia", "boa tarde", "boa noite"]


def ajuda(texto: str) -> bool:
    chaves = [
        "ajuda",
        "o que você faz",
        "o que voce faz",
        "como funciona"
    ]
    return any(c in texto.lower() for c in chaves)


def responder_llm(pergunta: str) -> str:
    entrada = prompt.format(pergunta=pergunta)
    resposta = llm.invoke(entrada)

    if "Resposta:" in resposta:
        resposta = resposta.split("Resposta:")[-1]

    resposta = limpar_texto(resposta)

    if not resposta:
        return "Não consegui gerar uma resposta adequada."

    return resposta


# =========================================================
# AGENTE PRINCIPAL
# =========================================================

def agente(comando: str) -> str:
    comando = limpar_texto(comando)

    if not comando:
        return "Digite um comando."

    if saudacao(comando):
        return "Olá! Sou um agente inteligente criado com LangChain."

    if ajuda(comando):
        return (
            "Eu posso responder perguntas e fazer cálculos.\n\n"
            "Exemplos:\n"
            "• Quanto é 8 vezes 9?\n"
            "• Calcule 20% de 500\n"
            "• Qual é a raiz quadrada de 144?\n"
            "• Qual é a data de hoje?\n"
            "• Que dia foi ontem?\n"
            "• Daqui a 10 dias que data será?\n"
            "• O que é inteligência artificial?"
        )

    if pergunta_data(comando):
        resposta_data = interpretar_data_relativa(comando)
        if resposta_data:
            return resposta_data
        return data_atual.invoke({})

    if eh_matematica(comando):
        expressao = normalizar_expressao(comando)

        if expressao.startswith("sqrt(") and expressao.endswith(")"):
            try:
                numero = float(expressao[5:-1])
                return f"Resultado: {math.sqrt(numero)}"
            except Exception:
                return "Não consegui calcular a raiz quadrada."

        return calculadora.run(expressao)

    return responder_llm(comando)


# =========================================================
# LOOP PRINCIPAL
# =========================================================

if __name__ == "__main__":
    print("\n=== AGENTE INTELIGENTE (LangChain + HuggingFace) ===")
    print("Digite 'sair' para encerrar.\n")

    while True:
        comando = input("Você: ")

        if comando.lower() == "sair":
            print("Agente: Encerrando...")
            break

        resposta = agente(comando)
        print(f"Agente: {resposta}\n")
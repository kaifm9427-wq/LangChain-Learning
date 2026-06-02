from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
import os

from langchain_huggingface import (
    HuggingFaceEndpoint,
    ChatHuggingFace
)

load_dotenv()

model1 = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
)

hf_llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
    max_new_tokens=256,
    temperature=0.5,
    provider="auto"
)

model2 = ChatHuggingFace(llm=hf_llm)

prompt1 = PromptTemplate(
    template="generate short and simple notes from the following text\n{text}",
    input_variables=["text"]
)

prompt2 = PromptTemplate(
    template="generate 5 short question answer from the following text\n{text}",
    input_variables=["text"]
)

prompt3 = PromptTemplate(
    template="""
merge the provided notes and quiz into a single document

Notes:
{notes}

Quiz:
{quiz}
""",
    input_variables=["notes", "quiz"]
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    "notes": prompt1 | model1 | parser,
    "quiz": prompt2 | model2 | parser
})

merge_chain = prompt3 | model1 | parser

chain = parallel_chain | merge_chain

text = """
Reading books is a valuable habit that helps people gain knowledge, improve vocabulary, and develop critical thinking skills. Books allow readers to explore topics in depth and learn new ideas.

One major benefit of reading is improved concentration. When reading, people focus on a single topic for a longer time, which helps strengthen attention span.

Books can inspire personal growth and motivate people to achieve their goals. They also help reduce stress.

Overall, reading books is a simple activity that supports learning, creativity, and personal development.
"""

result = chain.invoke({"text": text})

print(result)

chain.get_graph().print_ascii()


from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)
#this model doesn't give by default structured output
model=ChatHuggingFace(llm=llm)

#1st prompt->detailed report
template1=PromptTemplate(
    template="Write a detailed report on {topic}",
    input_variables=['topic']
)

#2nd prompt->5 line summary
template2=PromptTemplate(
    template="write a 5 line summary on the following text report./n {text}",
    input_variables=['text']
)

parser=StrOutputParser()

chain=template1 | model | parser | template2 | model | parser
result=chain.invoke({'topic':'black hole'})

print(result)
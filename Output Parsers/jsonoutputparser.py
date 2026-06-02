from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)
#this model doesn't give by default structured output
model=ChatHuggingFace(llm=llm)

parser=JsonOutputParser()

template=PromptTemplate(
    template="""you must return ONLY valid JSON.{format_instructions} give me the name , age, city of fictional person.\n {format_instructions}""",
    input_variables=[],
    partial_variables={'format_instructions':parser.get_format_instructions()}
)

chain=template | model | parser
result=chain.invoke({})
print(result)

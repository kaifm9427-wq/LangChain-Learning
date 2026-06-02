from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain.output_parsers.structured import StructuredOutputParser, ResponseSchema
load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)
#this model doesn't give by default structured output
model=ChatHuggingFace(llm=llm)

schema=[
    ResponseSchema(name='fact_1', description='fact 1 about the topic')
    ResponseSchema(name='fact_2', description='fact 2 about the topic')
    ResponseSchema(name='fact_3', description='fact 3 about the topic')
]
parser=StructuredOutputParser.from_response_schemas(schema )

template=PromptTemplate(
    template='give three facts about topic{topic} \n {format_instruction}',
    input_variables=['topic'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

prompt=template.invoke({'topic':'black hole'})
result=model.invoke(prompt)
final_result=parser.parse(result.content)
print(final_result)

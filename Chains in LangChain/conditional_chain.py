from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnableLambda

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
    max_new_tokens=256,
    temperature=0,
    provider="auto"
)

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

class Feedback(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(
        description="Give the sentiment of the feedback."
    )

parser2 = PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(
    template="Classify the sentiment of the following feedback text into positive or negative.\n\n{feedback}\n\n{format_instruction}",
    input_variables=["feedback"],
    partial_variables={
        "format_instruction": parser2.get_format_instructions()
    }
)

classifier_chain = prompt1 | model | parser2

prompt_positive = PromptTemplate(
    template="Respond to this positive feedback in one sentence.\n\nFeedback: {feedback}",
    input_variables=["feedback"]
)

prompt_negative = PromptTemplate(
    template="Respond to this negative feedback in one sentence.\n\nFeedback: {feedback}",
    input_variables=["feedback"]
)

branch_chain = RunnableBranch(
    (
        lambda x: x["sentiment"].sentiment == "positive",
        prompt_positive | model | parser
    ),
    (
        lambda x: x["sentiment"].sentiment == "negative",
        prompt_negative | model | parser
    ),
    RunnableLambda(lambda x: "Could not determine sentiment.")
)

chain = (
    RunnableParallel(
        sentiment=classifier_chain,
        feedback=RunnableLambda(lambda x: x["feedback"])
    )
    | branch_chain
)

print(chain.invoke({"feedback": "This is a terrible phone"}))
print(chain.invoke({"feedback": "This phone is amazing"}))
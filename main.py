import argparse

from dotenv import load_dotenv
from langchain.llms.openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain


load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="return a list of numbers")
parser.add_argument("--language", default="python")
args = parser.parse_args()


llm = OpenAI()

# Chain A
code_prompt = PromptTemplate(template="Write a very short {language} function that will {task}", input_variables=["language", "task"])
code_chain = LLMChain(
    llm=llm,
    prompt=code_prompt,
    output_key="code"
)

# Chain B
test_prompt = PromptTemplate(template="Write a unit test for the following {language} code:\n{code}", input_variables=["language", "code"])
test_chain = LLMChain(
    llm=llm,
    prompt=test_prompt,
    output_key="test"
)

chain = SequentialChain(
    chains=[code_chain, test_chain],
    input_variables=["language", "task"],
    output_variables=["test", "code"]
)


result = chain({
    "language": args.language,
    "task": args.task
})

print(">>>>>>>> GENERATED CODE: ")
print(result["code"])

print(">>>>>>>> GENERATED TEST: ")
print(result["test"])
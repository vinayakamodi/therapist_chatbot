import boto3
from langchain_aws import ChatBedrock

def initialize_bedrock_client():
    """
    Initialize and return a ChatBedrock instance with preconfigured model parameters.
    """
    bedrock = boto3.client('bedrock-runtime', region_name='us-east-1')
    model_kwargs = {
        "temperature": 0,
        "top_k": 250,
        "top_p": 1,
        "max_tokens_to_sample": 2**10
    }
    model_id = 'anthropic.claude-3-5-sonnet-20241022-v2:0'
    return ChatBedrock(client=bedrock, model_id=model_id, model_kwargs=model_kwargs)

def get_response(text):
    """
    Invoke the Bedrock model with the given text and return the response.
    """
    llm = initialize_bedrock_client()
    response = llm.invoke(text)
    return response

def main():
    """
    A simple main function to test Bedrock model invocation.
    """
    print(get_response("tell me a joke"))

if __name__ == "__main__":
    main()

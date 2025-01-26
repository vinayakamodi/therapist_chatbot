import boto3
from botocore.exceptions import ClientError
import os
import sys
from getpass import getpass

# Set AWS credentials as environment variables
os.environ["AWS_ACCESS_KEY_ID"] = "" # Use from the mail
os.environ["AWS_SECRET_ACCESS_KEY"] = "" # Use from the mail

def create_bedrock_client():
    """
    Create and return a Bedrock Runtime client in the specified AWS Region.
    """
    return boto3.client("bedrock-runtime", region_name="us-east-1")

def run_chatbot():
    """
    Run the chatbot that continuously listens for user input, sends it to the configured model,
    and prints the model's response until the user decides to quit.
    """
    client = create_bedrock_client()

    # Model configuration
    model_id = "meta.llama3-8b-instruct-v1:0"
    max_tokens = 512
    system_message = "[INST][INST]You are a very amazing therapist with exceptional emotional intelligence.[/INST]"
    conversation = []

    while True:
        try:
            user_message = input("\nEnter a prompt (empty to quit): ").strip()
            if user_message:
                conversation.append({
                    "role": "user",
                    "content": [{"text": system_message + user_message}],
                })

                response = client.converse(
                    modelId=model_id,
                    messages=conversation,
                    inferenceConfig={"maxTokens": max_tokens, "temperature": 0.5, "topP": 0.9},
                )

                response_text = response["output"]["message"]["content"][0]["text"]
                print(response_text + "\n")

                conversation.append({
                    "role": "assistant",
                    "content": [{"text": response_text}],
                })
            else:
                print("[Exited]")
                break

        except (ClientError, Exception) as e:
            print(f"ERROR: Can't invoke '{model_id}'. Reason: {e}")
            sys.exit(1)

if __name__ == "__main__":
    run_chatbot()

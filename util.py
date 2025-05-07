import os
import json
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

def get_access_token(var):

    access_token=None

    try:

        with open("token.json", "r") as token_file:
            config = json.load(token_file)
            access_token = config[var]

    except Exception as e:

        print(f"Unable to obtain access token")
        raise

    return access_token

def set_env(var):

    access_token = get_access_token(var)
    if access_token:
        os.environ[var] = access_token

def get_analyst_prompts(max_analysts:int,user_query:str,consolidated_outline:str):

    try:
        
        sys_prompt=""
        with open("./prompts/generate_analysts/sys_prompt.txt","r",encoding='utf-8') as file:

            sys_prompt = file.read()

        usr_prompt=""
        with open("./prompts/generate_analysts/usr_prompt.txt","r",encoding='utf-8') as file:

            usr_prompt = file.read()

    except Exception as e:

        print(f"Something went wrong while reading prompts {e}")
        raise

    sys_prompt = sys_prompt.format(max_analysts=max_analysts)
    usr_prompt = usr_prompt.format(max_analysts=max_analysts,topic=user_query,related_themes=consolidated_outline)

    messages=[]

    sys_message = SystemMessage(content=sys_prompt)
    usr_message = HumanMessage(content=usr_prompt)

    messages = [sys_message, usr_message]

    return messages

def get_interview_transcript(messages):

    transcript=""
    for message in messages:
        if(message.name=="expert"):
            transcript += "Expert: "+message.content+"\n"
        else:
            transcript += f"{message.name}: "+message.content+"\n"

    return transcript
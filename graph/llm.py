from langchain_openai import ChatOpenAI



def get_llm(model_name,
            num_requests=1,
            temperature=1.0,
            tools=None,
            tool_choice="auto",
            parallel_tool_calls=False,
            response_format=None):


    llm = ChatOpenAI(
                model_name = model_name,
                temperature = temperature,
                max_tokens = 5000,
                n = num_requests,
            )


    return llm
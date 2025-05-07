from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.runnables.config import RunnableConfig
from tavily import TavilyClient
from pydantic import BaseModel, Field, ConfigDict
from langgraph.types import StreamWriter, Send
from langgraph.graph import END, MessagesState
from langchain_core.messages import get_buffer_string
from typing import List
import os
import textwrap

from graph.llm import get_llm
from util import get_interview_transcript
from graph.state import (WebSearchState, ContentState, OutlineState, GenerateAnalystsState, 
Analyst, InterviewState, ResearchGraphState)

class LLM:

    def __init__(self,args,output_format=None,tools=None):

        self.llm = get_llm(
                model_name=args["model_name"],
                num_requests=args["num_requests"],
                temperature=args["temperature"]
            )

        if(tools):
            self.llm = self.llm.bind_tools(
                                tools=tools,
                                tool_choice="auto",
                                strict=True,
                                parallel_tool_calls=False
                                )

        if(output_format):
            self.llm = self.llm.with_structured_output(output_format)    

    def __call__(self,messages):

        response = self.llm.invoke(messages)

        return response

## Final Report
class Report(BaseModel):
    report: str = Field(description="The final report written by LLM.")
    model_config = ConfigDict(extra='forbid')

def write_report(state: ResearchGraphState,config:RunnableConfig, writer:StreamWriter):

    # Full set of sections
    sections = state["sections"]
    topic = state["topic"]

    # Concat all sections together
    formatted_str_sections = ""
    for indx,section in enumerate(sections):
        formatted_str_sections += "Interview report "+str(indx+1)+" :\n\n"
        formatted_str_sections += section + "\n\n"

    try:

        sys_prompt=""
        with open("./prompts/final_report/sys_prompt.txt","r",encoding='utf-8') as file:

            sys_prompt = file.read()

    except Exception as e:

        print(f"Something went wrong while reading prompts {e}")
        raise

    # Summarize the sections into a final report
    system_message = sys_prompt.format(topic=topic, context=formatted_str_sections)
    system_message = SystemMessage(content=system_message)

    llm_call = LLM(config["configurable"]["input_args"],Report)

    response = llm_call([system_message])

    return {"final_report": response.report}

def initiate_interviews(state:ResearchGraphState, config:RunnableConfig, writer:StreamWriter):

    temp_state =  {
                    "messages": [AIMessage(f"So you said you were writing an article on {state["topic"]}?", name="expert")],
                    "topic": state["topic"], 
                    "outline": state["outline"],
                    "max_num_turns": 3,
                  }
    
    calls=[]
    for analyst in state["analysts"]:

        state = {**temp_state, **{"analyst":analyst}}
        calls.append(Send("conduct_interview", state))

    return calls

## Interview

def conduct_interview(state:InterviewState, config:RunnableConfig, writer:StreamWriter):

    from graph.graph import build_interview_graph

    input_args = config["configurable"]["input_args"]
    interview_subgraph = build_interview_graph()

    interview_state = interview_subgraph.invoke(state,config=config)

    return {"sections":[interview_state["section"]]}

class SearchQuery(BaseModel):
    search_query: str = Field(description="Search query for web search")
    model_config = ConfigDict(extra='forbid')

class Section(BaseModel):
    title: str = Field(description="Title for summary generated.")
    summary: str = Field(description="The summary of interview written by LLM.")
    model_config = ConfigDict(extra='forbid')

def write_section(state: InterviewState, config:RunnableConfig, writer:StreamWriter=None):

    """ Node to answer a question """

    try:

        sys_prompt=""
        with open("./prompts/interview/section_writer/sys_prompt.txt","r",encoding='utf-8') as file:

            sys_prompt = file.read()

    except Exception as e:

        print(f"Something went wrong while reading prompts {e}")
        raise

    # Get state
    interview = state["interview"]
    context = "\n".join(state["context"])
    analyst = state["analyst"]
   
    system_message = sys_prompt.format(topic=state["topic"],persona=analyst.persona,
                                        interview=state["interview"])
    system_message = SystemMessage(content=system_message)

    llm_call = LLM(config["configurable"]["input_args"],Section)

    response = llm_call([system_message])

    section = "Title: "+response.title+"\n"
    section += response.summary

    print("Section:")
    print(section)
                
    return {"section": section}

def save_interview(state: InterviewState, config:RunnableConfig, writer:StreamWriter=None):
    
    """ Save interviews """

    # Get messages
    messages = state["messages"]
    
    # Convert interview to a string
    interview = get_interview_transcript(messages)
    
    # Save to interviews key
    return {"interview": interview}

def route_interview(state:InterviewState, config:RunnableConfig, writer:StreamWriter=None):

    """ Route between question and answer """
    
    # Get messages
    messages = state["messages"]
    max_num_turns = state.get('max_num_turns',2)

    # Check the number of expert answers 
    num_responses = len(
        [m for m in messages if isinstance(m, AIMessage) and m.name == "expert"]
    )

    # End if expert has answered more than the max turns
    if num_responses >= max_num_turns:
        return "save_interview"

    # This router is run after each question - answer pair 
    # Get the last question asked to check if it signals the end of discussion
    last_question = messages[-2]
    
    if "Thank you so much for your help" in last_question.content:
        return "save_interview"

    return "generate_question_interview"

def generate_question_interview(state:InterviewState, config:RunnableConfig, writer:StreamWriter=None):
    """ Node to generate a question """

    try:

        sys_prompt=""
        with open("./prompts/interview/generate_question/sys_prompt.txt","r",encoding='utf-8') as file:

            sys_prompt = file.read()

    except Exception as e:

        print(f"Something went wrong while reading prompts {e}")
        raise

    # Get state
    analyst = state["analyst"]
    messages = state["messages"]
    topic = state["topic"]

    topics_assigned = "\n".join(analyst.themes_allocated)

    # Generate question 
    system_message = sys_prompt.format(name=analyst.name,profession=analyst.role,
                                    description=analyst.description, topic=topic, 
                                    topics_interested=topics_assigned,outline=state["outline"])

    system_message = SystemMessage(content=system_message)

    llm_call = LLM(config["configurable"]["input_args"])

    question = llm_call([system_message]+messages)

    question.name = analyst.name.replace(" ","_")

    print(analyst.name)
    print(question.content)
        
    return {"messages": question}

def search_web_interview(state:InterviewState, config:RunnableConfig, writer:StreamWriter=None):
    # Transform the given user query.

    try:

        sys_prompt=""
        with open("./prompts/interview/query_reformulation/sys_prompt.txt","r",encoding='utf-8') as file:

            sys_prompt = file.read()

    except Exception as e:

        print(f"Something went wrong while reading prompts {e}")
        raise

    sys_message = SystemMessage(content=sys_prompt)

    llm_call = LLM(config["configurable"]["input_args"],SearchQuery)

    response = llm_call([sys_message]+state["messages"])

    client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

    response = client.search(
                    query=response.search_query,
                    search_depth="advanced",
                    max_results=3,
                    chunks_per_source = 3, #number of content blocks for each source.
                    include_answer=True,
                    include_raw_content=False
                )
    
 
    return {"context":[response["answer"]]}


def answer_question_interview(state:InterviewState, config:RunnableConfig, writer:StreamWriter=None):

    try:

        sys_prompt=""
        with open("./prompts/interview/answer_question/sys_prompt.txt","r",encoding='utf-8') as file:

            sys_prompt = file.read()

    except Exception as e:

        print(f"Something went wrong while reading prompts {e}")
        raise

    # Get state
    analyst = state["analyst"]
    messages = state["messages"]
    context_ls = state["context"]

    context = "\n".join(context_ls)

    sys_prompt = sys_prompt.format(context=context)
    sys_message = SystemMessage(content=sys_prompt)

    llm_call = LLM(config["configurable"]["input_args"])

    answer = llm_call([sys_message]+messages)
    answer.name = "expert"

    print("Expert:")
    print(answer.content)
        
    return {"messages": answer}


## Generate analysts

class Perspectives(BaseModel):
    analysts: List[Analyst] = Field(
        description="Comprehensive list of analysts with their roles and affiliations.",
    )
    model_config = ConfigDict(extra='forbid')

def feedback_analyst(state:GenerateAnalystsState, config:RunnableConfig):

    feedback = input("Do you want to give feedback? Type your feedback or 'NO' to finish: ")
    
    return {"feedback":feedback.strip()}
    

def generate_analysts(state:GenerateAnalystsState, config:RunnableConfig):

    feedback = state.get("feedback",None)

    if(feedback):
        human_feedback = HumanMessage(content="Following is feedback:\n"+feedback)
        state["messages"].append(human_feedback)

    llm_call = LLM(config["configurable"]["input_args"],Perspectives)

    response = llm_call(state["messages"])

    generated_analysts= "\n".join([analyst.persona + ("-" * 80) for analyst in response.analysts])

    print(generated_analysts)

    """
    generated_analysts=""
    if response.analysts:
        for analyst in response.analysts:
            generated_analysts += "\n".join([
                                        f"Name: {analyst.name}",
                                        f"Affiliation: {analyst.affiliation}",
                                        f"Role: {analyst.role}",
                                        f"Description: {analyst.description}"])
            generated_analysts += "\n"+("-" * 50)
    """  

    ai_message = AIMessage(content="Following are analysts:\n"+generated_analysts)

    return {"analysts":response.analysts,"messages":ai_message}

def continue_feedback_analyst(state:GenerateAnalystsState, config:RunnableConfig):

    if(state["feedback"].upper()!="NO"):
        return "generate_analysts"

    return END


## Outline Agent nodes

class GenerateOutline(BaseModel):
    reasoning : str = Field(description= "Reasoning of how the outline was generated.")
    outline : str = Field(description="The outline produced by the LLM.")
    model_config = ConfigDict(extra='forbid')    

def consolidate_outlines(state:OutlineState, config:RunnableConfig, writer:StreamWriter):

    outlines=""
    for indx,outline in enumerate(state["outlines"]):
        outlines += str(indx+1) + ". " + outline + "\n\n"

    try:
        
        sys_prompt=""
        with open("./prompts/consolidate_outline/sys_prompt.txt","r",encoding='utf-8') as file:

            sys_prompt = file.read()

        usr_prompt=""
        with open("./prompts/consolidate_outline/usr_prompt.txt","r",encoding='utf-8') as file:

            usr_prompt = file.read()

    except Exception as e:

        print(f"Something went wrong while reading prompts {e}")
        raise

    usr_prompt = usr_prompt.format(topic=state["user_query"],outlines=outlines)

    messages=[]

    sys_message = SystemMessage(content=sys_prompt)
    usr_message = HumanMessage(content=usr_prompt)

    messages = [sys_message, usr_message]

    llm_call = LLM(config["configurable"]["input_args"],GenerateOutline)

    response = llm_call(messages)

    #writer({"consolidate_outline":response.outline})

    return {"consolidated_outline":response.outline}


def websearch_subgraph(state:OutlineState, config:RunnableConfig, writer:StreamWriter):

    from graph.graph import build_websearch_graph

    input_args = config["configurable"]["input_args"]
    websearch_subgraph = build_websearch_graph()

    #for event in websearch_subgraph.stream({"user_query":state["user_query"]}, config=config,stream_mode="custom"):
    #    print(event)
    response = websearch_subgraph.invoke({"user_query":state["user_query"]}, config=config)

    return {"contents":response["contents"]}

def create_outline(state:ContentState, config:RunnableConfig, writer:StreamWriter):

    try:
        
        sys_prompt=""
        with open("./prompts/outline_formation/sys_prompt.txt","r",encoding='utf-8') as file:

            sys_prompt = file.read()
        
        example_txt=""
        with open("./prompts/outline_formation/example.txt","r",encoding='utf-8') as file:

            example_txt = file.read()

        usr_prompt=""
        with open("./prompts/outline_formation/user_prompt.txt","r",encoding='utf-8') as file:

            usr_prompt = file.read()

    except Exception as e:

        print(f"Something went wrong while reading prompts {e}")
        raise

    sys_prompt = sys_prompt.format(example=example_txt)
    usr_prompt = usr_prompt.format(topic=state["user_query"],raw_content=state["content"])

    messages=[]

    sys_message = SystemMessage(content=sys_prompt)
    usr_message = HumanMessage(content=usr_prompt)

    messages = [sys_message, usr_message]

    llm_call = LLM(config["configurable"]["input_args"],GenerateOutline)

    response = llm_call(messages)

    #writer({"outline":response.outline})

    return {"outlines":[response.outline]}

def continue_to_outlines(state:OutlineState, config:RunnableConfig, writer:StreamWriter):

    
    return [Send("create_outline",{"user_query":state["user_query"],"content":content}) for content in state["contents"]]


### Web Search Agent nodes

class QueryTransform(BaseModel):
    user_query : str = Field(description="User input query.")
    search_string : str = Field(description="User query optimized for web search.")
    model_config = ConfigDict(extra='forbid')

def query_transform(state:WebSearchState, config:RunnableConfig, writer:StreamWriter=None):
    # Transform the given user query.

    try:

        sys_prompt=""
        with open("./prompts/query_reformulation/sys_prompt.txt","r",encoding='utf-8') as file:

            sys_prompt = file.read()

    except Exception as e:

        print(f"Something went wrong while reading prompts {e}")
        raise

    messages=[]

    sys_message = SystemMessage(content=sys_prompt)
    usr_message = HumanMessage(content="User query:\n\n"+state["user_query"])

    messages = [sys_message, usr_message]

    llm_call = LLM(config["configurable"]["input_args"],QueryTransform)

    response = llm_call(messages)

    return {"search_string":response.search_string}

def tavily_search(state:WebSearchState, config:RunnableConfig, writer:StreamWriter=None):
    # We can also use tavily search as a tool.

    client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

    response = client.search(
                    query=state["search_string"],
                    search_depth="advanced",
                    max_results=3,
                    chunks_per_source = 3, #number of content blocks for each source.
                    include_answer=True,
                    include_raw_content=True
                )

    results = response["results"]

    contents=[]
    for result in results:
        raw_content = result["raw_content"]
        contents.append(raw_content)
        #writer({"tavily_search":result['url']})

    return {"contents":contents}
from util import set_env, get_analyst_prompts
from graph.graph import (build_websearch_graph, build_outline_graph, build_analyst_graph,
build_interview_graph, build_final_report)
from langchain_core.messages import AIMessage
from args import get_args


def research(args):

    
    input_args = {
            "model_name":"gpt-4o",
            "num_requests":1,
            "temperature":0.2
    }

    user_query = args.topic
    max_analysts = args.max_analysts

    set_env("OPENAI_API_KEY")
    set_env("TAVILY_API_KEY")

    config = {"configurable": {"thread_id": "1", "input_args":input_args}}
    outline_graph = build_outline_graph()

    outline_state = outline_graph.invoke({"user_query":user_query}, config=config)

    print(outline_state["consolidated_outline"])

    analysts_messages = get_analyst_prompts(max_analysts,user_query,outline_state["consolidated_outline"])

    analyst_graph = build_analyst_graph()

    analysts_state = analyst_graph.invoke(
                                        {
                                        "messages": analysts_messages,
                                        "user_query": user_query, 
                                        "outline": outline_state["consolidated_outline"],
                                        "max_analysts": max_analysts
                                        },
                                        config=config
                                        )
    """
    #test interview graph
    interview_graph = build_interview_graph()

    interview_state = interview_graph.invoke(
                                            {
                                            "messages": [AIMessage(f"So you said you were writing an article on {user_query}?", name="expert")],
                                            "topic": user_query, 
                                            "outline": outline_state["consolidated_outline"],
                                            "max_num_turns": 3,
                                            "analyst": analysts_state["analysts"][0]
                                            },
                                            config=config
                                            )
    """

    report_graph = build_final_report()

    report_state = report_graph.invoke(
                                        {
                                        "topic": user_query, 
                                        "outline": outline_state["consolidated_outline"],
                                        "analysts": analysts_state["analysts"]
                                        },
                                        config=config
                                    )

    print("Final Report: ")
    print("--------------------------------------------------------------------------------------")
    print(report_state["final_report"])

    output_file = user_query+".txt"
    with open(f"./outputs/{output_file}", "w", encoding='utf-8') as file:
        file.write(report_state["final_report"])

if __name__== "__main__":

    args = get_args()
    research(args)
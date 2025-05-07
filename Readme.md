

# 🧠 (STORM-based) AI Research Assistant 

This repository contains the implementation of an agentic research assistant that automates the process of generating comprehensive, Wikipedia-style articles on user-defined topics. The system employs the STORM (Synthesis of Topic Outlines through Retrieval and Multi-perspective Question Asking) methodology, developed by Stanford University, to enhance the depth and breadth of generated content. This repo utilizes the LangGraph framework.

## 📌 Features

* **Automated Outline Generation**: Creates a structured outline of subtopics and themes related to the user's query.
* **Analyst Persona Simulation**: Generates multiple analyst personas, each focusing on different aspects of the topic.
* **Expert Interviews**: Conducts simulated interviews between analyst personas and AI-generated experts to gather diverse perspectives.
* **Comprehensive Report Synthesis**: Consolidates insights from all interviews into a coherent, well-structured final report.

## 🔄 Workflow

1. **Outline Creation**: Given a user query, the system generates a detailed outline comprising various subtopics and themes.
2. **Analyst Generation**: For each theme in the outline, the system creates an analyst persona with specific interests and expertise.
3. **Conducting Interviews**: Each analyst conducts a simulated interview with an AI-generated expert, focusing on their assigned themes.
4. **Report Consolidation**: The system synthesizes information from all interviews to produce a comprehensive final report on the original topic.

## 🧰 Technologies Used

* **Python**: Core programming language for implementation.
* **LangChain & LangGraph**: Frameworks for building and managing the conversational agents and workflows.
* **OpenAI GPT Models**: Utilized for generating outlines, personas, interview questions, and synthesizing reports.
* **Tavily API**: Searches the web via Tavily API calls.


## 🚀 Getting Started

### Prerequisites

* Python 3.8 or higher
* OpenAI API Key
* Tavily API Key
* Required Python packages (listed in `requirements.txt`)

### Installation

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/saketd403/research-assistant.git
   cd research-assistant
   ```

2. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Set Up Environment Variables**:
   Create a `.env` file in the root directory and add your OpenAI API key:

   ```env
   OPENAI_API_KEY=your_openai_api_key
   TAVILY_API_KEY=your_tavily_key
   ```

### Usage

Run the main script with your desired topic:

```bash
python main.py --topic "Your Research Topic Here" --max_analysts "Total number of interviewers"
```

The system will generate the outline, simulate analyst interviews, and produce the final report, which will be saved in the `outputs/` directory.

## 📄 Example

Input:

```
"Impacts of Climate Change on Coastal Ecosystems"
```

Output:

* `outputs/Impacts of Climate Change on Coastal Ecosystems.txt`: Consolidated report synthesizing all gathered information.

## 🤝 Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your enhancements.

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 📚 References

* [STORM: Assisting in Writing Wikipedia-like Articles From Scratch with Large Language Models](https://arxiv.org/abs/2402.14207)
* [LangGraph STORM Tutorial](https://langchain-ai.github.io/langgraph/tutorials/storm/storm/)

---

Feel free to customize this `README.md` to better fit the specifics of your project, such as updating the repository URL, adjusting the technologies used, or adding more detailed instructions. If you need assistance with any of these modifications or further customization, please let me know!


🛠️ To Do
 - Add support for multiple reflection patterns to reduce hallucination and improve factual grounding of LLM responses.
 - Add reflection patterns as guardrails whereever necessary
 - Add an LLM that does spelling checks and correct grammatical errors.
 - Add more sources for search e.g. perform search using google scholar, use reddit, etc.
 - Add a way to rank the webpages on how well they could answer the question asked.
 - Instead of using tavily' answer, use and LLM to produce a comprehensive summary based on the webpage
   to answer interview questions. 


Note:

When an interviewer poses a question, the system performs a web search using Tavily to gather relevant context. To optimize for efficiency and stay within the language model's context window limitations, only the summarized answer from Tavily is provided to the expert for response generation. This approach helps prevent exceeding token limits and ensures consistent performance.

While it's possible to include the full raw_content from Tavily's search results, doing so may lead to context window overflows, especially with longer documents. If you choose to include full content, be mindful of the potential impact on processing and token limits.
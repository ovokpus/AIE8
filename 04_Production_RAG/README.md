<p align = "center" draggable=”false” ><img src="https://github.com/AI-Maker-Space/LLM-Dev-101/assets/37101144/d1343317-fa2f-41e1-8af1-1dbb18399719" 
     width="200px"
     height="auto"/>
</p>

## <h1 align="center" id="heading">Production RAG with LangGraph and LangChain</h1>

| 📰 Session Sheet | ⏺️ Recording     | 🖼️ Slides        | 👨‍💻 Repo         | 📝 Homework      | 📁 Feedback       |
|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|
| [Session 4: RAG with LangGraph, OSS Local Models, & Eval w/ LangSmith ](https://www.notion.so/Session-4-Production-Grade-RAG-with-LangChain-and-LangSmith-26acd547af3d80838d5beba464d7e701) | Coming soon! | Coming soon! | You are here! | Coming soon! | Coming soon! | Coming soon!

# Build 🏗️

If running locally:

1. `uv sync`
2. **NEW**: Set up Ollama by running the `Ollama_Setup_and_Testing.ipynb` notebook first to verify your local LLM setup
3. Open the main assignment notebook
4. Select the venv created by `uv sync` as your kernel

## Setting up Ollama (Local LLM)

Before starting the main assignment, run the `Ollama_Setup_and_Testing.ipynb` notebook to:
- Verify Ollama is installed and running
- Test embeddings with LangChain connectors
- Test model inference with LangChain connectors
- Ensure all models are properly downloaded

Run the preparation notebook and complete the contained tasks:

- 🤝 Breakout Room #1:
    1. Install and run Ollama
    2. Ensure all the required models are pulled
    3. Test them!

Next, run the Assignment notebook and complete the contained tasks:

- 🤝 Breakout Room #2:
    1. (Optional) Setup LangSmith tracing
    2. Understanding LangGraph States and Nodes
    3. Implementing a Simple RAG Graph
    4. Start to think about extending the Graph with Complex Flows

# Ship 🚢

- The completed notebook. 
- 5min. Loom Video

# Share 🚀
- Walk through your notebook and explain what you've completed in the Loom video
- Make a social media post about your final application and tag @AIMakerspace
- Share 3 lessons learned
- Share 3 lessons not learned

# Submitting Your Homework

Follow these steps to prepare and submit your homework:
1. Create a branch of your `AIE7` repo to track your changes. Example command: `git checkout -b s04-assignment`
2. Responding to the activities and questions in both the `Assignment_Introduction_to_LCEL_and_LangGraph_LangChain_Powered_RAG.ipynb`and `LangSmith_and_Evaluation` notebooks:
    + Option 1: Provide your responses in a separate markdown document:
      + Create a markdown document in the `04_Production_RAG` folder of your assignment branch (for example “ACTIVITIES_QUESTIONS.md”):
      + Copy the activities and questions into the document
      + Provide your responses to these activities and questions
    + Option 2: Respond to the activities and questions inline in the notebooks:
      + Edit the markdown cells of the activities and questions then enter your responses
      + NOTE: Remember to create a header (example: `##### ✅ Answer:`) to help the grader find your responses
3. Add (if you created a separate document), commit, and push your responses to your `origin` repository.

> _NOTE on the `Assignment_Introduction_to_LCEL_and_LangGraph_LangChain_Powered_RAG` notebook: You will also need to enter your response to Question #1 in the code cell directly below it which contains this line of code:_
    ```
    embedding_dim =  # YOUR ANSWER HERE
    ```

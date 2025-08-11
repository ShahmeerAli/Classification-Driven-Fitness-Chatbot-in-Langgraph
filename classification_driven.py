from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from typing import Annotated,List,Literal,TypedDict
from langchain_core.messages import HumanMessage,SystemMessage,AIMessage,BaseMessage
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from pydantic import BaseModel,Field
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph,END
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv,find_dotenv
import os
import pandas as pd



load_dotenv(find_dotenv())

GROQ_API_KEY=os.environ['GROQAPI_KEY']




embedding_function=HuggingFaceEmbeddings()

llm=ChatGroq(
    model="llama3-70b-8192",
    groq_api_key=os.environ.get("GROQAPI_KEY")
)

df = pd.read_excel(r"D:\LangGraph\07_RAG\gym.xlsx")

str_df=df.to_string(index=False)
csv_doc=Document(
    page_content=str_df,
    metadata={
        "source":"gym.xlsx"
    }
)

docs=[]
pdf_docs=[]
pdf_files = [
    r"D:\LangGraph\07_RAG\one.pdf",
    r"D:\LangGraph\07_RAG\two.pdf",
    r"D:\LangGraph\07_RAG\three.pdf",
    r"D:\LangGraph\07_RAG\five.pdf"
]

for doc in pdf_files:
    loader=PyPDFLoader(doc)
    pdf_docs.extend(loader.load())

docs = [csv_doc] + pdf_docs

db=Chroma.from_documents(docs,embedding_function)

retriever=db.as_retriever(search_type="mmr",search_kwargs={"k":3})


prompt_template="""
  ANSWER THE QUESTION BASED ONLY ON TEH FOLLOWING CONTEXT: {context}
  QUESTION:{question}
"""

prompt=ChatPromptTemplate.from_template(prompt_template)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain=prompt | llm

class AgentState(TypedDict):
    messages:list[BaseMessage]
    documents:list[Document]
    ontopic:str


class GradeQuestion(BaseModel):
    """
    Bool value to check whether the question is from gym or not
    """
    score:str=Field(
        description="Question is  about gym ? if yes -> 'Yes if not ->No"
    )


def question_classifier(state: AgentState): 
    question = state["messages"][-1].content
    system = """ You are a classifier that determines whether a user's question is about one of the following topics 
    
    1. Comprehensive Exercises Guide
    2. Daily and Weekly Health Routines (detailed)
    3. Benefits of Running and Cardio (in-depth)
    4. Gyms — Pros, Cons, and Alternatives (analysis)
    5. Exercise Risks, Modifications, and Comparisons (detailed)
    6. Facilities & Equipment
    
    If the question IS about any of these topics, respond with 'Yes'. Otherwise, respond with 'No'.

    """

    grade_prompt=ChatPromptTemplate.from_messages(
      [
        ("system",system),
        ("human","user_question : {question}")
      ]
    )    

    structure_llm=llm.with_structured_output(GradeQuestion)
    grade_llm=grade_prompt | structure_llm
    result=grade_llm.invoke({"question":question})
    state["ontopic"]=result.score
    return state



def on_Topic_router(state:AgentState):
    on_topic=state["ontopic"]
    if on_topic=="Yes":
        return "on_topic"
    else:
        return "off_topic"


def retriever_docs(state:AgentState):
    question=state['messages'][-1].content
    documents=retriever.invoke(question)
    state['documents']=documents
    return state


def generate_answer(state:AgentState):
    question=state['messages'][-1].content
    documents=state['documents']
    generation=rag_chain.invoke(
        {
            "context":documents,"question":question
        }
    )
    state['messages'].append(generation)


def off_topic(state:AgentState):
    state["messages"].append(AIMessage(
        content="I am sorry this topic is not related to Fitness"
    ))


graph=StateGraph(AgentState)

graph.add_node("topic_decision",question_classifier)
graph.add_node("off_topic_response",off_topic)
graph.add_node("retrieve",retriever_docs)
graph.add_node("generate_answer",generate_answer)

graph.add_conditional_edges(
    "topic_decision",
    on_Topic_router,
    {
        "on_topic":"retrieve",
        "off_topic":"off_topic_response"
    }
)

graph.add_edge("retrieve","generate_answer")
graph.add_edge("generate_answer",END)
graph.add_edge("off_topic_response",END)


graph.set_entry_point("topic_decision")
app=graph.compile()               

response=app.invoke(
    input={
        "messages":[HumanMessage(
            content="What are the benefits of Running?"
        )]
    }
)

print(response)


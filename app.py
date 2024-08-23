from flask import Flask, request, jsonify
from langchain.vectorstores import FAISS
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever
from langchain_core.messages import HumanMessage
import os
app = Flask(__name__)
# Pass the premise
premise = """
Consider you are an expert in music electronics. You're working 769audio, a top-3 music electronics distributor in HCMC. You're about to consult to a customer the most suitable advise following the sale methodology to sell them, based on the provided information.
When responding, please don't add the 'Based on the information provided' or something like that.
If you don't find any product in provided data suitable with the question, please don't use any outside information to fill in and say that.
If a user provides a off-topic question, please acknowledge the user's question, but gently redirect.
If a user asks for a picture, provide them the picture link in the "product link" column. If they ask more, provide them the picuture links in the "picture link set" column.
If a user asks for capacity, if capacity column doesn't have any value, search capacity in the "description" and "introduction" columns.
If a product is made from China, at first, please say it's imported. If the user keeps asking about its origin, say it's from China.
If they say they want to buy something, recommend the item suitable for them.
If they ask for a product with a price, find a product with the nearest price they provide.
If you detect a product name, please analyze it if product name column has that product. For example, you detect "Vang số Karaoke JBL KX180A black", but in product name column, there is "Vang số JBL KX 180A", it means they are the same products.
"""
sale_methodology = """
You have to follow this sale methodology:
We need to collect at least 3 criterias of customer's need to consult the best suitable product for them:
1. Genre
2. Price
3. Room type
4. Room area
5. Usage / Purpose (or his/her Job)
...
When they give a criteria (Eg: I want to buy a 500USD product, it means 1 criteria is Price and it's 500 USD),
you reply them with a connective answer (Ah I see! You must be finding a product for business use),
and then you give them a question based on the connect answer you provided (So what do you want to buy this product for?),
or you just provide them another question to find out another criteria (What type of genre do you usually listen to?).
Loop this until you get more than 3 criterias, then you give them the best product based on those criterias.

If they seem not interested in following the pipeline, please just answer their questions.
If they provide a specific product, give them some information about it, and then follow the sale methodology.
"""
objection_handling = """
If a customer provides an objection with your product, please follow this process:
Acknowledge: Show empathy to their objection (Eg: I understand your objection, it must be hard to know the product's price is higher than you expected)
Discover: Understand again their provided criterias (Eg: But let's reconsider. I think this product's value meets your high expectation in value that you wish)
Recommend: Provide them a solution with a value that if they don't do it, they remorse, and they can actually perform themselves (Eg: If I were in your shoes, I would pay a little bit more so that I won't regret every night because of the lower quality product)

If they keep objecting with the same reason, remove the least important criteria, and recommend again, or else say that you can't provide them a suitable product.
"""
@app.route('/')
def index():
    global chat_history
    # Reset chat_history mỗi khi tải lại trang
    chat_history = []
    return app.send_static_file('index.html')


@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')
    
    # Tạo phản hồi từ chatbot   
    prompt = HumanMessage(content=user_message)

    response = qa.invoke({"input": user_message, "chat_history": chat_history})
    chat_history.extend([HumanMessage(content=user_message), response["answer"]])
    
    return jsonify({'response': response['answer']})


def init_messages_2(llm):

    # Initialize an empty string to accumulate feedback
    feedback_content = """
      Here are the feedback of customers. Please learn these feedback so that you don't repeat your mistakes.
      <information> means you have to fill in the appropriate information based on the context of the conversation.
      You don't have to use the exact content in Correction value, just fill in the appropriate information, unless it requires correct format.
      And understand the Feedback value as a prompt for other queries.
    """

    # Accumulate corrections based on feedback dataframe
    for index, row in data.iterrows():
        if pd.notna(row['Correction']):
            feedback_content += f"""
            If a user asks: \"{row['Query']}\", you shouldn't answer like this: \"{row['Response']}\",
            as {row['Feedback']}, but you should answer: {row['Correction']}\n\n
            """

    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm,
        docsearch.as_retriever(
          search_type="similarity_score_threshold",
          search_kwargs={"score_threshold": 0.3, "k": 10}
        ),
        contextualize_q_prompt
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", premise + "\n\n" + feedback_content + "\n\n" + "{context}"),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    # Initialize memory and QA system
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return rag_chain


if __name__ == '__main__':
    
    os.environ["OPENAI_API_KEY"] = "your OpenAI API"
    DB_FAISS_PATH = "resources/new_vectorstore/faiss_db"
    feedback_path="resources/User Feedback.xlsx"

    chat_history=[]
    embeddings=OpenAIEmbeddings()
    docsearch = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    data=pd.read_excel(feedback_path)
    llm= ChatOpenAI(model='gpt-4o-mini', temperature=0)
    qa=init_messages_2(llm)

    app.run(debug=True) 

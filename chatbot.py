from flask import Flask, request, jsonify, session
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import sys
import pandas as pd
from langchain.chat_models import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever
from langchain.schema import SystemMessage
from langchain_core.messages import HumanMessage
import os
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.docstore.document import Document

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


def retrieve_and_filter_chunks(row_numbers, data, excluded_columns=["Giới thiệu", "Chi tiết"]):
    filtered_chunks = []
    for row_number in row_numbers:
        # Check if row number is valid before accessing
        if row_number in data.index:
            row_data = data.loc[row_number]
            for col in data.columns:
                if col not in excluded_columns:
                    filtered_chunks.append(
                        Document(page_content=str(row_data[col]),
                                 metadata={"source": col,
                                           "row": row_number}))
    return filtered_chunks


def retrieve_and_combine_documents(query, data, retriever, chat_history):
    initial_docs = retriever.invoke(query, chat_history=chat_history)
    row_numbers = {doc.metadata["row"] for doc in initial_docs}  # Using a set comprehension to ensure uniqueness

    # If you need the result as a list
    row_numbers = list(row_numbers)
    filtered_docs = retrieve_and_filter_chunks(row_numbers, data)
    filtered_docs.extend(initial_docs)

    return filtered_docs


def custom_retrieval_logic(query, chat_history, data, retriever):
    return retrieve_and_combine_documents(query, data, retriever, chat_history)


def initialize_rag(llm, data, retriever):
    def wrapped_retriever(input_data):
        # print("Contextualized prompt:", input_data.content)

        input_query = input_data.content
        chat_history = ""

        return custom_retrieval_logic(input_query, chat_history, data, retriever)

    feedback_content = """
      Here are the feedback of customers. Please learn these feedback so that you don't repeat your mistakes.
      Learn the correct format after "as the feedback is" so that you can apply the format for other similar questions.
      <information> means you have to fill in the appropriate information based on the context of the conversation.
      You don't have to use the exact content in Correction value, just fill in the appropriate information, unless it requires correct format.
    """

    # Accumulate corrections based on feedback dataframe
    for index, row in feedback_df.iterrows():
        if pd.notna(row['Correction']):
            feedback_content += f"""
            If a user asks: \"{row['Query']}\", you shouldn't answer like this: \"{row['Response']}\",
            as the feedback is {row['Feedback']}, but you should answer: {row['Correction']}\n\n
            """

    contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""

    input_premise = """
Bạn là một chuyên gia về thiết bị điện tử âm nhạc. Bạn đang làm việc tại 769audio, một trong ba nhà phân phối thiết bị âm nhạc hàng đầu tại TP.HCM. Bạn sắp tư vấn cho một khách hàng về phương pháp bán hàng phù hợp nhất để bán hàng cho họ, dựa trên thông tin được cung cấp.
    """

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", input_premise + "\n\n" + contextualize_q_system_prompt), # + feedback_content + "\n\n"
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    # Create a history-aware retriever using the custom wrapped retriever
    history_aware_retriever = contextualize_q_prompt | llm | wrapped_retriever

    qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", premise + "\n\n" + feedback_content + "\n\n" + "{context}"),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

    # Initialize memory and QA system
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    # Create and return the RAG chain
    return create_retrieval_chain(history_aware_retriever, question_answer_chain)


if __name__ == '__main__':
    os.environ["OPENAI_API_KEY"] = "your OpenAI API"
    DB_FAISS_PATH = "resources/new_vectorstore/faiss_db"
    feedback_path = "resources/User Feedback.xlsx"
    data_path = "resources/Finished_Data_in_769audio_vn.csv"

    embeddings = OpenAIEmbeddings()
    docsearch = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    feedback_df = pd.read_excel(feedback_path)
    data = pd.read_csv(data_path)
    llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)
    retriever = docsearch.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "score_threshold": 0.3,
            "k": 20
        }
    )

    chat_history = []
    qa = initialize_rag(llm, data, retriever)

    app.run(debug=True)

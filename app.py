
import streamlit as st
import pandas as pd
from chat import RAGChatBot

def initialize_bot():
    if 'chatbot' not in st.session_state:
        bot = RAGChatBot()
        df = pd.read_csv('demo4.csv')
        bot.documents = df['text'].tolist()
        bot.subject = 'Chatbot tư vấn tuyển sinh'
        bot.db = bot.create_chroma_db()
        st.session_state.chatbot = bot
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []

def display_chat_history():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def main():
    st.set_page_config(
        page_title="HCMUTE Tuyển Sinh Chatbot",
        page_icon="🎓"
    )

    st.title("🎓 HCMUTE - Trợ lý tư vấn tuyển sinh")
    st.markdown("""
    Xin chào! Tôi là trợ lý ảo tư vấn tuyển sinh của trường Đại học Sư phạm Kỹ thuật TP.HCM (HCMUTE).
    Hãy đặt câu hỏi, tôi sẽ giúp bạn tìm hiểu thông tin về trường.
    """)

    initialize_bot()
    display_chat_history()

    if prompt := st.chat_input("Nhập câu hỏi của bạn..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get bot response
        bot = st.session_state.chatbot
        bot.query = prompt
        relevant_docs = bot.get_relevant_passage()
        
        # Display relevant documents in expander
        with st.expander("📚 Tài liệu tham khảo"):
            st.markdown(relevant_docs)

        # Generate and display response
        prompt_text = bot.make_prompt()
        response = bot.model.generate_content(prompt_text)

        with st.chat_message("assistant"):
            st.markdown(response.text)
        
        st.session_state.messages.append({"role": "assistant", "content": response.text})

if __name__ == "__main__":
    main()

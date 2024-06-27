import streamlit as st
from crewai import Agent, Task, Crew, Process
from langchain.agents import Tool
from langchain.tools import DuckDuckGoSearchRun
from langchain_community.utilities import SerpAPIWrapper
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
import os
from dotenv import load_dotenv
from langchain.utilities import SerpAPIWrapper
from langchain_community.chat_models import ChatOpenAI
# .env 파일에서 환경 변수를 로드합니다.
load_dotenv()

# 환경 변수에서 API 키를 가져옵니다.
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
os.environ['SERPAPI_API_KEY'] = os.getenv('SERPAPI_API_KEY')

# 검색 도구를 설정합니다.
search = DuckDuckGoSearchRun()
serpapi = SerpAPIWrapper()

# Agent가 사용할 도구들을 정의합니다.
tools = [
    Tool(name="DuckDuckGo Search", func=search.run, description="유용한 웹 검색 도구입니다."),
    Tool(name="SerpAPI", func=serpapi.run, description="Google 검색 결과에 접근할 수 있는 도구입니다.")
]

# GPT-4와 GPT-3.5-turbo 모델을 설정합니다.
gpt4_model = ChatOpenAI(model_name="gpt-4")
gpt3_5_turbo_model = ChatOpenAI(model_name="gpt-3.5-turbo")

# 대화 기록과 종료 상태를 세션 상태에 초기화합니다.
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'conversation_ended' not in st.session_state:
    st.session_state.conversation_ended = False

# Streamlit UI의 제목을 설정합니다.
st.title("Custom Crew AI Application")

# Agent 설정 입력 섹션을 만듭니다.
st.header("Agent 설정")
num_agents = st.number_input("Agent 수", min_value=1, max_value=5, value=3)

agents = []
for i in range(num_agents):
    st.subheader(f"Agent {i+1}")
    role = st.text_input(f"역할 {i+1}", value=f"Agent {i+1}")
    goal = st.text_input(f"목표 {i+1}", value="목표를 입력하세요")
    backstory = st.text_area(f"배경 이야기 {i+1}", value="배경 이야기를 입력하세요")
    
    # 마지막 Agent (최고 선임)에게 GPT-4 모델 할당, 나머지에게 GPT-3.5-turbo 할당
    model = gpt4_model if i == num_agents - 1 else gpt3_5_turbo_model
    
    # Agent 객체를 생성하고 리스트에 추가합니다.
    agent = Agent(
        role=role,
        goal=goal,
        backstory=backstory,
        verbose=True,
        allow_delegation=True,
        tools=tools,
        llm=model
    )
    agents.append(agent)

# Task 설정 입력 섹션을 만듭니다.
st.header("Task 설정")
num_tasks = st.number_input("Task 수", min_value=1, max_value=5, value=3)

tasks = []
for i in range(num_tasks):
    st.subheader(f"Task {i+1}")
    description = st.text_area(f"설명 {i+1}", value="Task 설명을 입력하세요")
    expected_output = st.text_area(f"예상 출력 {i+1}", value="예상되는 출력 형식을 입력하세요")
    agent_index = st.selectbox(f"담당 Agent {i+1}", range(num_agents))
    
    # Task 객체를 생성하고 리스트에 추가합니다.
    task = Task(
        description=description,
        expected_output=expected_output,
        agent=agents[agent_index]
    )
    tasks.append(task)

# 실시간 대화 내용을 표시할 컨테이너를 생성합니다.
chat_container = st.empty()

# Crew 실행 버튼을 만듭니다.
if st.button("Crew 실행"):
    # Crew 객체를 생성합니다.
    crew = Crew(
        agents=agents,
        tasks=tasks,
        verbose=2,
        process=Process.sequential
    )
    
    with st.spinner("Crew가 작업 중입니다..."):
        # 대화 내용을 저장할 리스트를 초기화합니다.
        conversation = []
        
        def process_output(output):
            # 출력을 한국어로 번역합니다.
            translated_output = gpt4_model.invoke([
                SystemMessage(content="You are a translator. Translate the following text to Korean:"),
                HumanMessage(content=output)
            ]).content
            conversation.append(translated_output)
            # 대화 내용을 업데이트하여 표시합니다.
            chat_container.text_area("대화 내용", value="\n\n".join(conversation), height=400)
        
        # CrewAI의 출력을 가로채서 처리하는 함수를 정의합니다.
        def custom_print(*args, **kwargs):
            # end 매개변수를 처리합니다. 기본값은 줄바꿈입니다.
            end = kwargs.get('end', '\n')
            # 모든 인자를 문자열로 변환하고 결합합니다.
            output = ' '.join(str(arg) for arg in args) + end
            process_output(output)
        
        # CrewAI의 print 함수를 오버라이드합니다.
        import builtins
        original_print = builtins.print
        builtins.print = custom_print
        
        try:
            # Crew를 실행하고 결과를 받습니다.
            result = crew.kickoff()
        finally:
            # 원래의 print 함수로 복원합니다.
            builtins.print = original_print
        
        # 최종 결과를 처리합니다.
        process_output(result)
    
    # 대화 내용을 세션 상태에 저장합니다.
    st.session_state.messages.extend(conversation)

# 대화 기록을 표시합니다.
st.header("전체 대화 기록")
for message in st.session_state.messages:
    st.write(message)

# 사용자 입력을 받고 응답하는 로직을 구현합니다.
user_input = st.chat_input("메시지를 입력하세요")
if user_input and not st.session_state.conversation_ended:
    st.session_state.messages.append(f"사용자: {user_input}")
    
    # AI의 응답을 생성합니다 (한국어로).
    response = gpt4_model.invoke([
        SystemMessage(content="You are a helpful AI assistant. Please respond in Korean."),
        HumanMessage(content=user_input)
    ])
    
    ai_response = response.content
    st.session_state.messages.append(f"AI: {ai_response}")
    
    # 대화 내용을 업데이트하여 표시합니다.
    chat_container.text_area("대화 내용", value="\n\n".join(st.session_state.messages), height=400)

# 대화 종료 버튼을 만듭니다.
if st.button("대화 종료") and not st.session_state.conversation_ended:
    st.session_state.conversation_ended = True
    
    # 최고 선임 Agent를 선택합니다.
    senior_agent = agents[-1]
    
    # 최종 요약 작업을 생성합니다.
    end_task = Task(
        description="대화를 종료하고 지금까지의 내용을 요약한 보고서를 작성해주세요. 한국어로 작성해주세요.",
        agent=senior_agent
    )
    
    # 요약을 위한 Crew를 생성하고 실행합니다.
    end_crew = Crew(
        agents=[senior_agent],
        tasks=[end_task],
        verbose=2,
        process=Process.sequential
    )
    
    end_message = end_crew.kickoff()
    
    st.session_state.messages.append(f"최종 요약: {end_message}")
    
    # 상세 보고서 작성 작업을 생성합니다.
    report_task = Task(
        description="지금까지의 대화 내용을 바탕으로 상세한 보고서를 작성해주세요. 한국어로 작성해주세요.",
        agent=senior_agent
    )
    
    # 보고서 작성을 위한 Crew를 생성하고 실행합니다.
    report_crew = Crew(
        agents=[senior_agent],
        tasks=[report_task],
        verbose=2,
        process=Process.sequential
    )
    
    report = report_crew.kickoff()
    
    st.header("최종 보고서")
    st.write(report)
    
    # 대화 내용을 업데이트하여 표시합니다.
    chat_container.text_area("대화 내용", value="\n\n".join(st.session_state.messages), height=400)

# 대화 내용 저장 기능을 구현합니다.
if st.button("대화 내용 저장"):
    chat_history = "\n".join(st.session_state.messages)
    st.download_button(
        label="대화 내용 다운로드",
        data=chat_history,
        file_name="chat_history.txt",
        mime="text/plain"
    )

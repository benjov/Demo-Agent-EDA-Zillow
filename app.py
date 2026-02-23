# Exploratory Data Analysis (EDA) Copilot App
# -----------------------
# streamlit run app.py

import os
import sys
import importlib
import html
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from dotenv import load_dotenv
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_openai import ChatOpenAI

from ai_data_science_team.ds_agents import EDAToolsAgent
from ai_data_science_team.utils.matplotlib import matplotlib_from_base64
from ai_data_science_team.utils.plotly import plotly_from_dict

# --- Compatibilidad LangChain v1 para librerías antiguas ---
try:
    sys.modules["langchain.prompts"] = importlib.import_module("langchain_core.prompts")
except Exception:
    pass
# -----------------------------------------------------------

# =============================================================================
# STREAMLIT APP SETUP (including data upload, API key, etc.)
# =============================================================================

# Carga .env (Streamlit no lo carga automáticamente)
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)

openai_api_key = os.getenv("OPENAI_API_KEY")

# Si no hay key, detén la app con un mensaje claro
if not openai_api_key:
    st.error("No se encontró OPENAI_API_KEY. Revisa tu archivo .env (misma carpeta que app.py) o configura la variable de entorno.")
    st.stop()

# Modelos más recientes (puedes ajustar esta lista a lo que tengas habilitado en tu cuenta)
# Nota: Mantengo un fallback con gpt-4o / gpt-4o-mini por compatibilidad.
MODEL_LIST = [
    "gpt-5-mini",
    "gpt-5",
#    "gpt-4o",
#    "gpt-4o-mini",
]

TITLE = "Tu Copiloto de Análisis Explotarorio de Datos y en Elaboración de Reportes"
st.set_page_config(page_title=TITLE, page_icon="📊")
st.title("📊 " + TITLE)

st.markdown(
    """
Bienvenido al Copiloto de Análisis de Propiedades. 
Este agente de IA está diseñado para ayudarte a analizar tus datos, y 
proporcionar informes que pueden utilizarse para comprender los datos.

Los datos DEMO son un listado de propiedades target del proyecto IA Realty
"""
)

with st.expander("Preguntas de Ejemplo", expanded=False):
    st.write(
        """
        - ¿Qué herramientas tienes disponibles? Devuelve una tabla.
        - Dame información sobre la tool X.
        - Explica el conjunto de datos.
        - ¿Qué contienen las primeras 5 filas?
        - Describe el conjunto de datos.
        - Selecciona las propiedades con mayor potencial de Flipping.
        """
    )

# Sidebar for file upload / demo data
st.sidebar.image("images/Logo_AB.png", use_container_width=True)
st.sidebar.markdown("Contact: vicente@analiticaboutique.com.mx & benjamin@analiticaboutique.com.mx")
st.sidebar.header("Copiloto de Análisis de Propiedades", divider=True)
use_demo_data = st.sidebar.checkbox("Usa datos demo", value=True)

if "DATA_RAW" not in st.session_state:
    st.session_state["DATA_RAW"] = None

if use_demo_data:
    demo_file_path = Path("data/Prediction_LASSO_DEMO.xlsx")
    if demo_file_path.exists():
        #df = pd.read_excel(demo_file_path)
        df = pd.read_excel(demo_file_path, dtype={"parcelId": "string"})
        file_name = "Zillow Miami Dataset"
        st.session_state["DATA_RAW"] = df.copy()
        st.write(f"## Preview of top 10 {file_name} data:")
        st.dataframe(st.session_state["DATA_RAW"].head(10))
    else:
        st.error(f"Demo data file not found at {demo_file_path}. Please ensure it exists.")

# =============================================================================
# OpenAI Model Selection (actualizado a modelos recientes)
# =============================================================================

model_option = st.sidebar.selectbox("Choose OpenAI model", MODEL_LIST, index=0)

# Si quieres forzar baja latencia / costo, puedes parametrizar temperature, max_tokens, etc.
# Mantengo un set mínimo y compatible.
OPENAI_LLM = ChatOpenAI(
    model=model_option,
    api_key=openai_api_key,
    #temperature=0.2,
)
llm = OPENAI_LLM

# =============================================================================
# CHAT MESSAGE HISTORY AND ARTIFACT STORAGE
# =============================================================================

msgs = StreamlitChatMessageHistory(key="langchain_messages")
if len(msgs.messages) == 0:
    msgs.add_ai_message("¿Cómo puedo ayudarte?")

if "chat_artifacts" not in st.session_state:
    st.session_state["chat_artifacts"] = {}

def display_chat_history():
    """
    Renders the entire chat history along with any artifacts attached to messages.
    Artifacts (e.g., plots, dataframes, Sweetviz reports) are rendered inside expanders.
    """
    for i, msg in enumerate(msgs.messages):
        with st.chat_message(msg.type):
            st.write(msg.content)
            if "chat_artifacts" in st.session_state and i in st.session_state["chat_artifacts"]:
                for artifact in st.session_state["chat_artifacts"][i]:
                    with st.expander(artifact["title"], expanded=True):
                        if artifact["render_type"] == "dataframe":
                            st.dataframe(artifact["data"])
                        elif artifact["render_type"] == "matplotlib":
                            st.pyplot(artifact["data"])
                        elif artifact["render_type"] == "plotly":
                            st.plotly_chart(artifact["data"])
                        elif artifact["render_type"] == "sweetviz":
                            report_file = artifact["data"].get("report_file")
                            try:
                                with open(report_file, "r", encoding="utf-8") as f:
                                    report_html = f.read()
                            except Exception as e:
                                st.error(f"Could not open report file: {e}")
                                report_html = "<h1>Report not found</h1>"
                            report_html_escaped = html.escape(report_html, quote=True)
                            html_code = f"""
                            <!DOCTYPE html>
                            <html>
                            <head>
                            <meta charset="utf-8">
                            <title>Sweetviz Report</title>
                            <style>
                                body, html {{
                                margin: 0;
                                padding: 0;
                                height: 100%;
                                }}
                                #iframe-container {{
                                position: relative;
                                width: 100%;
                                height: 600px;
                                }}
                                #myIframe {{
                                width: 100%;
                                height: 100%;
                                border: none;
                                }}
                                #fullscreen-btn {{
                                position: absolute;
                                top: 10px;
                                right: 10px;
                                z-index: 1000;
                                padding: 8px 12px;
                                background-color: #007bff;
                                color: white;
                                border: none;
                                border-radius: 4px;
                                cursor: pointer;
                                }}
                            </style>
                            </head>
                            <body>
                            <div id="iframe-container">
                                <button id="fullscreen-btn" onclick="toggleFullscreen()">Full Screen</button>
                                <iframe id="myIframe" srcdoc="{report_html_escaped}" allowfullscreen></iframe>
                            </div>
                            <script>
                                function toggleFullscreen() {{
                                var container = document.getElementById("iframe-container");
                                if (!document.fullscreenElement) {{
                                    container.requestFullscreen().catch(err => {{
                                    alert("Error attempting to enable full-screen mode: " + err.message);
                                    }});
                                    document.getElementById("fullscreen-btn").innerText = "Exit Full Screen";
                                }} else {{
                                    document.exitFullscreen();
                                    document.getElementById("fullscreen-btn").innerText = "Full Screen";
                                }}
                                }}
                                
                                document.addEventListener('fullscreenchange', () => {{
                                if (!document.fullscreenElement) {{
                                    document.getElementById("fullscreen-btn").innerText = "Full Screen";
                                }}
                                }});
                            </script>
                            </body>
                            </html>
                            """
                            components.html(html_code, height=620)
                        else:
                            st.write("Artifact of unknown type.")

# =============================================================================
# PROCESS AGENTS AND ARTIFACTS
# =============================================================================

def process_exploratory(question: str, llm, data: pd.DataFrame) -> dict:
    """
    Initializes and calls the EDA agent using the provided question and data.
    Processes any returned artifacts (plots, dataframes, etc.) and returns a result dict.
    """
    eda_agent = EDAToolsAgent(
        llm,
        invoke_react_agent_kwargs={"recursion_limit": 10},
    )

    question += " Don't return hyperlinks to files in the response."

    eda_agent.invoke_agent(
        user_instructions=question,
        data_raw=data,
    )

    tool_calls = eda_agent.get_tool_calls()
    ai_message = eda_agent.get_ai_message(markdown=False)
    artifacts = eda_agent.get_artifacts(as_dataframe=False)

    result = {
        "ai_message": ai_message,
        "tool_calls": tool_calls,
        "artifacts": artifacts
    }

    if tool_calls:
        last_tool_call = tool_calls[-1]
        result["last_tool_call"] = last_tool_call
        tool_name = last_tool_call

        print(f"Tool Name: {tool_name}")

        if tool_name == "explain_data":
            result["explanation"] = ai_message

        elif tool_name == "describe_dataset":
            if artifacts and isinstance(artifacts, dict) and "describe_df" in artifacts:
                try:
                    df = pd.DataFrame(artifacts["describe_df"])
                    result["describe_df"] = df
                except Exception as e:
                    st.error(f"Error processing describe_dataset artifact: {e}")

        elif tool_name == "visualize_missing":
            if artifacts and isinstance(artifacts, dict):
                try:
                    matrix_fig = matplotlib_from_base64(artifacts.get("matrix_plot"))
                    bar_fig = matplotlib_from_base64(artifacts.get("bar_plot"))
                    heatmap_fig = matplotlib_from_base64(artifacts.get("heatmap_plot"))
                    result["matrix_plot_fig"] = matrix_fig[0]
                    result["bar_plot_fig"] = bar_fig[0]
                    result["heatmap_plot_fig"] = heatmap_fig[0]
                except Exception as e:
                    st.error(f"Error processing visualize_missing artifact: {e}")

        elif tool_name == "correlation_funnel":
            if artifacts and isinstance(artifacts, dict):
                if "correlation_data" in artifacts:
                    try:
                        corr_df = pd.DataFrame(artifacts["correlation_data"])
                        result["correlation_data"] = corr_df
                    except Exception as e:
                        st.error(f"Error processing correlation_data: {e}")
                if "plotly_figure" in artifacts:
                    try:
                        corr_plotly = plotly_from_dict(artifacts["plotly_figure"])
                        result["correlation_plotly"] = corr_plotly
                    except Exception as e:
                        st.error(f"Error processing correlation funnel Plotly figure: {e}")

        elif tool_name == "generate_sweetviz_report":
            if artifacts and isinstance(artifacts, dict):
                result["report_file"] = artifacts.get("report_file")
                result["report_html"] = artifacts.get("report_html")

        else:
            if artifacts and isinstance(artifacts, dict):
                if "plotly_figure" in artifacts:
                    try:
                        plotly_fig = plotly_from_dict(artifacts["plotly_figure"])
                        result["plotly_fig"] = plotly_fig
                    except Exception as e:
                        st.error(f"Error processing Plotly figure: {e}")
                if "plot_image" in artifacts:
                    try:
                        fig = matplotlib_from_base64(artifacts["plot_image"])
                        result["matplotlib_fig"] = fig
                    except Exception as e:
                        st.error(f"Error processing matplotlib image: {e}")
                if "dataframe" in artifacts:
                    try:
                        df = pd.DataFrame(artifacts["dataframe"])
                        result["dataframe"] = df
                    except Exception as e:
                        st.error(f"Error converting artifact to dataframe: {e}")
    else:
        result["plain_response"] = ai_message

    return result

# =============================================================================
# MAIN INTERACTION: GET USER QUESTION AND HANDLE RESPONSE
# =============================================================================

if st.session_state["DATA_RAW"] is not None:
    question = st.chat_input("Enter your question here:", key="query_input")
    if question:
        with st.spinner("Thinking..."):
            msgs.add_user_message(question)
            result = process_exploratory(
                question,
                llm,
                st.session_state["DATA_RAW"]
            )

            tool_name = result.get("last_tool_call")

            ai_msg = result.get("ai_message", "")
            if tool_name:
                ai_msg += f"\n\n*Tool Used: {tool_name}*"

            msgs.add_ai_message(ai_msg)

            artifact_list = []
            if tool_name == "describe_dataset" and "describe_df" in result:
                artifact_list.append({
                    "title": "Dataset Description",
                    "render_type": "dataframe",
                    "data": result["describe_df"]
                })
            elif tool_name == "visualize_missing":
                if "matrix_plot_fig" in result:
                    artifact_list.append({
                        "title": "Missing Data Matrix",
                        "render_type": "matplotlib",
                        "data": result["matrix_plot_fig"]
                    })
                if "bar_plot_fig" in result:
                    artifact_list.append({
                        "title": "Missing Data Bar Plot",
                        "render_type": "matplotlib",
                        "data": result["bar_plot_fig"]
                    })
                if "heatmap_plot_fig" in result:
                    artifact_list.append({
                        "title": "Missing Data Heatmap",
                        "render_type": "matplotlib",
                        "data": result["heatmap_plot_fig"]
                    })
            elif tool_name == "correlation_funnel":
                if "correlation_data" in result:
                    artifact_list.append({
                        "title": "Correlation Data",
                        "render_type": "dataframe",
                        "data": result["correlation_data"]
                    })
                if "correlation_plotly" in result:
                    artifact_list.append({
                        "title": "Correlation Funnel (Interactive Plotly)",
                        "render_type": "plotly",
                        "data": result["correlation_plotly"]
                    })
            elif tool_name == "generate_sweetviz_report":
                artifact_list.append({
                    "title": "Sweetviz Report",
                    "render_type": "sweetviz",
                    "data": {"report_file": result.get("report_file"), "report_html": result.get("report_html")}
                })
            else:
                if "plotly_fig" in result:
                    artifact_list.append({
                        "title": "Plotly Figure",
                        "render_type": "plotly",
                        "data": result["plotly_fig"]
                    })
                if "matplotlib_fig" in result:
                    artifact_list.append({
                        "title": "Matplotlib Figure",
                        "render_type": "matplotlib",
                        "data": result["matplotlib_fig"]
                    })
                if "dataframe" in result:
                    artifact_list.append({
                        "title": "Dataframe",
                        "render_type": "dataframe",
                        "data": result["dataframe"]
                    })

            if artifact_list:
                msg_index = len(msgs.messages) - 1
                st.session_state["chat_artifacts"][msg_index] = artifact_list

# =============================================================================
# FINAL RENDER: DISPLAY THE COMPLETE CHAT HISTORY WITH ARTIFACTS
# =============================================================================

display_chat_history()

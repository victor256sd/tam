import streamlit as st
import streamlit_authenticator as stauth
import openai
from openai import OpenAI
from openai import AsyncOpenAI
from agents import Agent, ItemHelpers, MessageOutputItem, Runner, FileSearchTool, function_tool, trace
import asyncio
import os
import pandas as pd
import openpyxl
import tiktoken
import json
import time
import yaml
from yaml.loader import SafeLoader
from pathlib import Path
from PyPDF2 import PdfReader, PdfWriter
from PIL import Image
import pytesseract
from cryptography.fernet import Fernet
import re

# Wait until run process completion.
def wait_on_run(client, run, thread):
    while run.status == "queued" or run.status == "in_progress":
        run = client.beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=run.id,
        )
        time.sleep(0.5)
    return run

# Retrieve messages from the thread, including message added by the assistant.
def get_response(client, thread):
    return client.beta.threads.messages.list(thread_id=thread.id, order="asc")

# Start client, thread, create file and add it to the openai vector store, update an
# existing openai assistant with the new vector store, create a run to have the 
# assistant process the vector store.
def generate_response(filename, openai_api_key, model, assistant_id, query_text):    
    # Check file existence.
    if filename is not None:
        # Start client, thread.
        client = OpenAI(api_key=openai_api_key)
        thread = client.beta.threads.create()
        # Start thread.
        client.beta.threads.messages.create(
            thread_id=thread.id, role="user", content=query_text
        )
        
        # Create file at openai storage from the uploaded file.
        file = client.files.create(
            file=open(filename, "rb"),
            purpose="assistants"
        )
        
        # Create vector store for processing by assistant.
        vector_store = client.vector_stores.create(
            name="aitam"
        )
        # Obtain vector store and file ids.
        TMP_VECTOR_STORE_ID = str(vector_store.id)
        TMP_FILE_ID = str(file.id)
        # Add the file to the vector store.
        batch_add = client.vector_stores.file_batches.create(
            vector_store_id=TMP_VECTOR_STORE_ID,
            file_ids=[TMP_FILE_ID]
        )        
        # Update Assistant, pointed to the vector store.
        assistant = client.beta.assistants.update(
            assistant_id,
            tools=[{"type": "file_search"}],
            tool_resources={
                "file_search":{
                    "vector_store_ids": [TMP_VECTOR_STORE_ID]
                }
            }
        )
        # Create a run to have assistant process the vector store file.
        run = client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=assistant_id,
        )
        # Wait on the run to complete, then retrieve messages from the thread.
        run = wait_on_run(client, run, thread)
        messages = get_response(client, thread)
    return messages, TMP_FILE_ID, TMP_VECTOR_STORE_ID, client, run, thread

# Constructed similar to above, exempt no use of the assistant. This calls the 
# llm with a user's query about the vector store. The vector store is re-
# constructed since every action on streamlit runs through the entire code.
# Parts of this function can likely be broken down into other functions. 
# The code might also be restructured to avoid re-building the vector store.
def generate_response_noassist(filename, openai_api_key, model, query_text):    
    # Check file existence.
    if filename is not None:
        # Start client, thread.        
        client = OpenAI(api_key=openai_api_key)
        thread = client.beta.threads.create()
        # Create file at openai storage from the uploaded file.
        file = client.files.create(
            file=open(filename, "rb"),
            purpose="user_data"
        )
        # Create vector store for processing by assistant.
        vector_store = client.vector_stores.create(
            name="aitam"
        )
        # Obtain vector store and file ids.        
        TMP_VECTOR_STORE_ID = str(vector_store.id)
        TMP_FILE_ID = str(file.id)
        # Add the file to the vector store.                        
        batch_add = client.vector_stores.file_batches.create(
            vector_store_id=TMP_VECTOR_STORE_ID,
            file_ids=[TMP_FILE_ID]
        )
        # Get messages from client based on user query of the vector store.
        messages = client.responses.create(
            input = query_text,
            model = model,
            temperature = 1,
            tools = [{
                "type": "file_search",
                "vector_store_ids": [TMP_VECTOR_STORE_ID],
            }]
        )        
    return messages, TMP_FILE_ID, TMP_VECTOR_STORE_ID, client

# Called when the advisory agent system exceeds the max turn limit.
# Uses a single OpenAI API call to generate a synthesized response
# based on the perspectives of the advisory group.
def fallback_summary_request(model, vs_id, query_text):
    INSTRUCTION_ENCRYPTED = b'gAAAAABohtGfHcmOGHFRTsdWg0GtzwWFPathTsqYs87K2kr2siCM-sZ7WhLDNj1Nn39tYpktrByZSbCf8JakwTLupxkfJNDoET3aLhhp8kZIMQPNSsAtDN5vp48I6TeJjBYI7qMwtEI3Sa3RIF2W-_uZFtR2ee6PFEhvKtxa_84_CILAgsJ9Fy6KP1Fi6mwFTftYDnKydQRHQpBQX_YTgkjfZZ7eYNbdNLsHQApJ17yPkSGyP4CBk6ucbiIR8osMNTPis2vQZ2RrmsfLdMN7dDU7uhmW9YkVIl3tCmcKrMZnAnP-8p-BN2lIoKOn2iPxZjlZCwFYBYkFia3yopsh9_bR9mSdn3wiqkIjYjZwRRHWRkBnolzlSTVC5flMkp9YJY75n-wqdvvcrWSKjrSQRoE-dtaa2zl6msFeyLugD9TCk_XjDfMpBrCTUzUtA7raDQhOevlZtDbvWsbr_bQ-YNRVwlBm0oBhNXmBKmmN0wrIqA1iRd7cOM_XXZVf0LsirCTg_R-GpIqwE5Mfj8QVRxla2oXdpMGCMwfz1LF0gLxRupcRZI6u7zgBlG7dXHqrYLTgSXSlBC7knkWcVbI3FXVCb3B4jmTnZjbLbSZk7fv6lovglgoh-TU6fVue26SgxMWH8SHDRe3zaK7QWlp7xLvb0Ar7LWg01DsRuBU1sgn7aMX6et8et7BQSOPWrrD9TYNHUyWam_6PagGA0EYg0pt7Om4noWJx_EtYbh9SrVtJsqDFaOJvtgYEsjmu_8iCjB0aHiLRSpWji2aaeewXXVtMjtWJp9ZM0u_w9NJHh-LdvqTPWZLxvWpRJaeSkVl4ip_macM1oOBlvO81y7jWkNv_ivQLgRYXPNESBcrt71zc4XT_1alvvzShueycx8Je-k21bOlYZzZ3TCjUc93010h07Fr-JPiLYUEUE4Zui4FWh5Ogv0QwsTvhgr7pXFsLcpzyCaS8Jxp_Z_nmFyixpMcAri7XwBL0eh1js1pVNsfvqhw6UqCWOkbrnK1254z41nuEDChNOty9dydU_OYHw7a_Sm8no3c9IGoa-j5m-sP-8ES4NeXleNHD_gm8XsLYJv1o_3O4D5tGGq_Xtr94SWcE1klOGCva6f72TzyEfIs6UvcBde7rxTiY6s2OlOA1FypuP_A1pCEkh24xGd2MN6px-x2f-UiSOm5NJbCCJTG2gK69y6b_dYD-zflAd2pMn_F0YeKrMMU3qWh-ILjQ29yJaJf8Ri19XqgZyBgE62Z5GOlQiNXmIDwnvq03FWjTx5ySUAYflChe1wgikuNHNXBa-xFK7qV0ySP_szOQiYHRDrUKjiDKGT32JwiItVmeYkt_zE_Bf1A3HCcOFiQ3Hdvd_xVETOAxsxECn58kCPgwdPd1JlICXS03xseAlQyIx4OYs-2B50iBi6Mk83YHOPOKtQ1TGnpBpDeCrwZb1YFIsAuc5QmScreH_P8s847AW9WhoVH3DcVtuf4LEFsoDysZbL-zKkBM6NecuAyLxraoTy9Ayu6IZ0k='
    key = st.secrets['INSTRUCTION_KEY'].encode()
    f = Fernet(key)
    INSTRUCTION = f.decrypt(INSTRUCTION_ENCRYPTED).decode()

    # Get messages from client based on user query of the vector store.
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    response = client.responses.create(
        input = INSTRUCTION,
        model = model,
        temperature = 0.6,
        tools = [{
            "type": "file_search",
            "vector_store_ids": vs_id,
        }]
    )

    cleaned_response = re.sub(r'【.*?†.*?】', '', response.output[1].content[0].text)
    return cleaned_response

# Initiate AI assistant and create a run to have the assistant answer the user
# query. 
async def generate_response_cmte(model, vs_id, query_text):    
    # client = OpenAI(api_key=openai_api_key)
    # thread = client.beta.threads.create()
    # # Start thread.
    # client.beta.threads.messages.create(
    #     thread_id=thread.id, role="user", content=query_text
    # )

    INSTRUCTION1_ENCRYPTED = b'gAAAAABohrfKeKyi7cJNlxQFjKlgnElw4zs3DyDqmUrMelL84BbaB7fABpc5SdaBgk8LtQpqYfBSext5KVJ3IEMipmiQo68TKiE3U-CCVbkSrBlQ2soPLroxZe90vgmFwPSVzGLmkAgVGYu_Hfkj6JJOJz4_ILG9gJNYJZgZBbqcmTnxZiLRUIYhOctURCuVGyx_QupFwCaLxrA5UplM0EkBs41NflecvtbnPf-bh5FYqJ8POxkIjzxViZa10bemGn9G52_e5FcydRj0a7Odz99cKWRsHP8bwmlqEA5IwQ0llDQEL1AbmmPgY5PlXOJQjRjx3eNCF9IP'
    INSTRUCTION2_ENCRYPTED = b'gAAAAABohrikHHiMjsrUNWcfWOKRXxdS2GLyFvi89a1Dz0g4WbHnFesTWPYk-gjIicWWqFIK5ylBqVogkKRRatczEoV9koinYg8SCvshUMTy3N3VjDNs6uxJqHbbbP5yU-P0tWLT1oWmV5ij7THbedi1Ay_ksXLFXcf7RrcgS4PitvRcpTVqFzfbMQubAHlVQUq_QkV3hiFuicb7wMFKMNE3XlmXIPElUqt6msXIsooYQm7DsRZVDoQyw4DcPhwQyYY7ljYbISqnNNsCaVSLV7zmQ_srbrTdWmjnx2LnQaT29_QobRN2IG4='
    INSTRUCTION3_ENCRYPTED = b'gAAAAABohrl4WetHeGSipX0T2BV7PprVCiO42Vq77Ln_bfV9mxBGBEgFXYdSGee28ZO6E5nPqKo3Vtq8rMi5W_ixAJEC1NmGuw-dY5uIJjDbRuFQCUqru-M_mh6z3oaYXPoIz58cHrs2JyWUIXtJllsb16jE83Z26bjt1ihJxEaI-OPrp3B2bmrKNdtH-t-345PRjV3ozYqtIOdywQiOEb-nGBtb8JHoyBNZj5nVKS_3SZhNNzjHPOTLAMFPdnZTQCX-DO85AUgy5ygGpS-7I-nnxe-PgGaBw4KumKuN0x_suBtvi6sTdr0='
    INSTRUCTION4_ENCRYPTED = b'gAAAAABohroZYUOEDOJEk3uahy7f7YESV77xekvW1OT0HVgU0uhVj4xyvSvfsdw4EPfUGxuPHJAXEd9cAfHFPdCpw21oLr8H6rSsLt4lwJ8AjRa8Wr29V2trNH-1fV8bJs3j-NTfYLB_vomAT9SYtNa4wKQyxSDKi1Q7owZkJ4mM_qqkVa6dn9n0vfqYsHfMdMx9TFL9uUZeoTKyZx1VAzG_cvf9Hj1dDcMTQlKEku-BOIMdiPzFsRPMkV690KtFhoLChTAWNdOrVVjRrxLZxyQptVxp14egx3WUpf6dgWvl2-Sh5MpjDs8P6GaIcEjm5_OqZrd2kjXC1xGgoGvjzqAlna1umILRug=='
    INSTRUCTION_ORCH_ENCRYPTED = b'gAAAAABohsoPvOB9VD4AXKY7wctD5iW69QZTIs5n7RGySuirQBPPy__4qcyRwyMIZq2gJZZG_B3cTRbEoPCb2XRe8TmemLx0nQMkwfp5LR2zOeqN6u2sgfEX4vJG5XP9rOJP4Pn5Lgav1ADFaFCzPRQofJ5on6zLhyvv8hrDHygto85PvhKwHeL9WMgEATfKmX-NX3IpOrSFup-v5thWB-Ns21Dq1zNMj7XXpmKE7PvWEY7f5jhH8JaDVvaB0KWEUAP6kfmrjbUhI4yuf2RJRRCRZLrr_IZ60A7V6PSSZln5aNdJdea-DQ-auWlaAztU3tdM4UT2edXhNvpuOtacOayLi3RMKC-oN9HJo02onybR6E4I8Par_Gk_0U5BKmUa-LT6GutY6MXDEonYHk4oJFwHni6agQeldUkvWWC7s7RQfVh-NwUwRr9HeFSBpQjPUQN8ad0cE9EYpFAMLGUwUZXexHqrmO2w31p0nhQ9fYEyXjiEk9JBRGmBQHG3W1dD0xKpPVdVj4v9nmBB6QxOlQW-bMzCaBNZrP0i_2HniO8x3BnBaCcK4TwZPjjQXNlbX1wVWx6TMSwXcFG0k1B4NcL4SuF5aLo5mFymYYgIbBRWFQCNs6vw8EhXscCKb_Wo04yCiqXnlfN0cbY65JcXwkqap920nWPhxiaQBE3D311a-N3hcJozTR-VDsZcSw3mhL0BcTRqg8KuKsZ57ftZUXJTdtnYncNT_sUeEmNcJ7cwNZswEV35qd3Q5XeEcmKfIEDWdD-Qu_lOsw3Ac-qyFh9eInWoKPzRUaFbcUu401YZC4eB6KHit_8rb_LbQP7C8IIglV7qdd0Hrb0c_ea7ZWJ_VFbTbH2os9oT6JZTK7sEN3MKLrjbJL2Z9LQ8NusFjPke0u9PcmV4JA_aKbnQBd3dtfNRMsdEXLOBt64t8LtHqT4Q9MBDe7rIQny1FBCgVFYlPceYeVZGHgZqsQZTkAix0AYN8B1pd49Z9k7h4f7q-1ScM9kck_NDBmdK0830uRSJVEap4vZwThRaHimgd4fsk-s3fO8_FkVIoKunncnunIPI5s39vmrl1kdyHW1A6vi29PUQcYonq0LLPrgj5o4dIYk3ztARhhM6nIZSajzkOj6P69Es6bW3j17W3hP7LF2eKHWwsQf6xQBlTvUmn92TIJkglsuOhOj3c48AS3Yd309M6xKYrRA8_esSXDUzJ2T7c4-zBoPnMtl8DrNSF6dDfAf2sR9EsdnbhMDyFrIjqULdQLh9ZRD8B4iuGYO3vSwnq7e-jaZnfImhdBEJFmjqYGOtch_fXcZ85CAfQ8RgP3zdtIJlq84mjrU9pY_qQvzMolOl6r3evrIDcnZmjRzHQ0_dspPIP8FVY2lHbK8--W0jHZ4-1j1WxDAgJd_qwmTXyY4IGNBOkeTqAkHGuKjRAfjeD-MMn9Al2rllqNi_vKQr4ILxwdiuYXLjapDeM-p3RCDNLABakzyTBL1trDOw1kFuEJCdx8LsEl3TKIanBuupRAKh5Ix0l4Bw57D3uoUXj4kz_FqeE9lY_vTJUyREmssvHfXIHA=='
    INSTRUCTION_SYNTH_ENCRYPTED = b'gAAAAABohr0g6gjkPwcA5BASMqyS-4Uazd4t49AHk0KUmewH1mF5C38Srj1ydFL_QMKAhgY3BASEKZkhG9aIjoIyGwcV1sz1AXr4zGQIlKLFVZMDYqmWSN-PWGpJniSS_ixYjpM0XhFodSVSC_AHTnDP1YExG7LNnjQbig6FwuaQan7QxA_oNWwrgKVbic-pwem-cm2uw4hYj9NxY4GM3pmctk-Y-2ttjdCJSWUyH4CyZgXMext6XIsLx5Hl1bgLiSdILe-C0pwF4hOyZD0nb1O2WYzDHyMJBDzBv5pB8bCXBPAKj4uM-UY4qBpuabCTeV23ptJhm9LY4nfRrKyUdMmsmwwrkTYPvqFkwGVwEcJ8CSqORPUTZClTh6FphP6aGPjB8pq2ujpJ9KjZIZTqEKSOZ2vMtbaLsQ8W0D8YoBXedCH5VoanRXdU-ak_N7WWweLfK8IokZr635uNdf6ds5XAER5ftaKlGzi9175jFwyzdYjVjbuwL-Ew0pcXCYzICe3li5NiXJZzL0cFZePEvrHosarnhlyIb_jiUDZg-JNt95CZ7Gh6_uaLzY_KqKy1TkrjMWKS9L94glMg82f9dUWHCPySvpmYxcHA5CHJf-OhSkHuSmmnHXGhTxyUZ_CxTYzuRnrqByucz28IgPztjldHp-98dhGtiCuVMm0mF_nyQcbbZsihjSlJ7G_Xcc82KTqWZmpIQWe2N_oDbCcXfsfC0AkUf3SIsLjAB698rv0SH4ay0AAGjP5BxZ4jI5gwTlWmHdzZ7Wx1fFrnHZmCeIS0d2fm-wUVynnJbUxPTRfNKZg1hhSrH0_Ek5-ViJE3on2XFEAhsbpPMslC-YnQuZ5HMLR5OSHJ0nl22BYVNgpR6ldFxXLT-PRZQ456i2L5f9H4z9ewY9g8a_2jHJJRwXr7nIReaZH2m22oMwS7-G7T4OSnTfKFH8Y6n7TeDMeWN3zHWFzp2JbWxRfscEIT_c5lwUwLrtlrSx3kDMo9xNq0LqmC5pVRS3cd3QyuCb_WNEEN_R5YZc25rT-kpmzxKu0LuiorfR-D9XZF141xP7wRDFpAClI25CSMMlHXxysR1RZVXfDu8choSqD9DPJKkzQnfN7N2kJWSd10mwi8wI4M0RGH6fL_MCi8Vn3UQVb5aCc0nQ2F5_xdnyJP2XnVJSs9WT5GtyK5w_lC_mivgUsRlmrCYeRrKU57-zdXslYXOkjRL3aaT8JpdOF29l9wayMUIZjse-U_yLJzLXWJdi62ujdCVyoy6O4B6Bz00MoFPASXuW8s87HE7f_dc5QUVKgLkTSqf8VFPXsSFW1-nkWeF9TD_7K4GnAp3NsYsK7ylFqVo6lkDRufpn9aULfFWclYKCDOEGIrRuB7bRMYpjlFRKlNO_RoDMPT_XstJ_HpH16qJ-laqv3ldGwX4uhkAsYUvahyfSUkg9kT1q5rxWXK-qMUalZ2Hejy20O_swgphnXXEBMctObIVu8kWyU3L8kJkOuBnajQlOQMv0RsjzhhMLVfWu6GcM3W6U1KhZFq-WLOfk8wFJTx'
    key = st.secrets['INSTRUCTION_KEY'].encode()
    f = Fernet(key)
    INSTRUCTION1 = f.decrypt(INSTRUCTION1_ENCRYPTED).decode()
    INSTRUCTION2 = f.decrypt(INSTRUCTION2_ENCRYPTED).decode()
    INSTRUCTION3 = f.decrypt(INSTRUCTION3_ENCRYPTED).decode()
    INSTRUCTION4 = f.decrypt(INSTRUCTION4_ENCRYPTED).decode()
    INSTRUCTION_ORCH = f.decrypt(INSTRUCTION_ORCH_ENCRYPTED).decode()
    INSTRUCTION_SYNTH = f.decrypt(INSTRUCTION_SYNTH_ENCRYPTED).decode()
    
    assist1_agent = Agent(
        name="security_agent",
        instructions=INSTRUCTION1,
        handoff_description="The safety and security expert",
        tools=[
            FileSearchTool(
                max_num_results=3,
                vector_store_ids=vs_id,
            )
        ],
        model=model,
    )
    assist2_agent = Agent(
        name="hr_agent",
        instructions=INSTRUCTION2,
        handoff_description="HR and labor relations representative",
        tools=[
            FileSearchTool(
                max_num_results=3,
                vector_store_ids=vs_id,
            )
        ],
        model=model,
    )
    assist3_agent = Agent(
        name="legal_agent",
        instructions=INSTRUCTION3,
        handoff_description="The legal advisor",
        tools=[
            FileSearchTool(
                max_num_results=3,
                vector_store_ids=vs_id,
            )
        ],
        model=model,
    )
    assist4_agent = Agent(
        name="mental_health_agent",
        instructions=INSTRUCTION4,
        handoff_description="The mental health and wellness expert",
        tools=[
            FileSearchTool(
                max_num_results=3,
                vector_store_ids=vs_id,
            )
        ],
        model=model,
    )
    orchestrator_agent = Agent(
        name="orchestrator_agent",
        instructions=(INSTRUCTION_ORCH),
        tools=[
            assist1_agent.as_tool(
                tool_name="security_agent",
                tool_description="The safety and security expert",
            ),
            assist2_agent.as_tool(
                tool_name="hr_agent",
                tool_description="HR and labor relations representative",
            ),
            assist3_agent.as_tool(
                tool_name="legal_agent",
                tool_description="The legal advisor",
            ),
            assist4_agent.as_tool(
                tool_name="mental_health_agent",
                tool_description="The mental health and wellness expert",
            ),
        ],
        model=model,
    )
    synthesizer_agent = Agent(
        name="synthesizer_agent",
        instructions=(INSTRUCTION_SYNTH),
        model=model,
    )
    # # Run the entire orchestration in a single trace
    # with trace("Orchestrator evaluator"):
    #     orchestrator_result = await Runner.run(orchestrator_agent, query_text)
    #     synthesizer_result = await Runner.run(synthesizer_agent, orchestrator_result.to_input_list())

    # openai.api_key = openai_api_key
    # client = AsyncOpenAI()
    # client = OpenAI(api_key=openai_api_key)
    
    try:
        orchestrator_result = await Runner.run(orchestrator_agent, query_text)
        synthesizer_result = await Runner.run(synthesizer_agent, orchestrator_result.to_input_list())
        return synthesizer_result        
    except Exception as e:
        response = fallback_summary_request(model, vs_id, query_text)
        synthesizer_result = await Runner.run(synthesizer_agent, response.to_input_list())
        return synthesizer_result            

async def orchestrator_init(orchestrator_agent, synthesizer_agent, query_text):
    synthesizer_result = []
    # Run the entire orchestration in a single trace
    with trace("Orchestrator evaluator"):
        orchestrator_result = Runner.run(orchestrator_agent, query_text)
        # for item in orchestrator_result.new_items:
        #     if isinstance(item, MessageOutputItem):
        #         text = ItemHelpers.text_message_output(item)
                # if text:
                #     print(f"  - Text: {text}")
        synthesizer_result.append(await Runner.run(synthesizer_agent, orchestrator_result))
    return synthesizer_result.final_output 

# Delete file in openai storage and the vector store.
def delete_vectors(client, TMP_FILE_ID, TMP_VECTOR_STORE_ID):
    # Delete the file and vector store
    deleted_vector_store_file = client.vector_stores.files.delete(
        vector_store_id=TMP_VECTOR_STORE_ID,
        file_id=TMP_FILE_ID
    )
    deleted_vector_store = client.vector_stores.delete(
        vector_store_id=TMP_VECTOR_STORE_ID
    )

def extract_text_from_excel(uploaded_file):
    output_filename = "temp.txt"
    df = pd.read_excel(uploaded_file, engine='openpyxl')
    df['combined_text'] = df.apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
    json_string = df.to_json(path_or_buf=None)
    serialized_data = json.dumps(json_string, indent=4)
    # Write serialized data to a text file.
    with open(output_filename, "w") as file:
        file.write(serialized_data)
    file.close()
    return output_filename

def copy_pdf(uploaded_file):
    # Specify the input and output file paths
    input_pdf_path = uploaded_file
    output_pdf_path = "temp.pdf"
    # Read the input PDF
    reader = PdfReader(input_pdf_path)    
    # Create a writer object to write the copy
    writer = PdfWriter()
    # Add all pages from the input PDF to the writer
    for page in reader.pages:
        writer.add_page(page)
    # Write the copied content to the output file
    with open(output_pdf_path, "wb") as output_file:
        writer.write(output_file)    
    output_file.close()
    return output_pdf_path

def convert_image_to_pdf(uploaded_file):
    output_file = "temp.txt"
    # Open the image file
    image = Image.open(uploaded_file)
    # Extract text from the image using pytesseract
    extracted_text = pytesseract.image_to_string(image)
    # Write the copied content to the output file
    with open(output_file, "w") as file:
        file.write(extracted_text)
    file.close()
    return output_file
    
# Disable the button called via on_click attribute.
def disable_button():
    st.session_state.disabled = True        

# Definitive CSS selectors for Streamlit 1.45.1+
st.markdown("""
<style>
    div[data-testid="stToolbar"] {
        display: none !important;
    }
    div[data-testid="stDecoration"] {
        display: none !important;
    }
    div[data-testid="stStatusWidget"] {
        visibility: hidden !important;
    }
</style>
""", unsafe_allow_html=True)

# Load config file with user credentials.
with open("config.yaml") as file:
    config = yaml.load(file, Loader=SafeLoader)

# Initiate authentication.
authenticator = stauth.Authenticate(
    config['credentials'],
)

# Call user login form.
result_auth = authenticator.login("main")
    
# If login successful, continue to aitam page.
if st.session_state.get('authentication_status'):
    authenticator.logout('Logout', 'main')
    st.write(f'Welcome *{st.session_state.get('name')}* !')

    # Model list, Vector store ID, assistant IDs (one for initial upload eval, 
    # the second for follow-up user questions).
    MODEL_LIST = ["gpt-4o-mini", "gpt-4.1-nano", "gpt-4.1-mini", "gpt-4.1"] #, "o4-mini"]
    VECTOR_STORE_ID = st.secrets["VECTOR_STORE_ID"]
    MATH_ASSISTANT_ID = st.secrets["MATH_ASSISTANT_ID"]
    MATH_ASSISTANT2_ID = st.secrets["MATH_ASSISTANT2_ID"]
    
    # Set page layout and title.
    st.set_page_config(page_title="Threat AI", page_icon=":spider:", layout="wide")
    st.header(":spider: Threat AI")
    
    # Field for OpenAI API key.
    # api_key_input = st.text_input(
    #         "OpenAI API Key",
    #         type="password",
    #         placeholder="Paste your OpenAI API key here (sk-...)",
    #         help="You can get your API key from https://platform.openai.com/account/api-keys.",  # noqa: E501
    #         value=os.environ.get("OPENAI_API_KEY", None) or st.session_state.get("OPENAI_API_KEY", "")
    #     )
    # Save the API key to the st session.
    # st.session_state["OPENAI_API_KEY"] = api_key_input
    openai_api_key = st.secrets["OPENAI_API_KEY"]
    
    # Retrieve user-selected openai model.
    model: str = st.selectbox("Model", options=MODEL_LIST)
    
    # Create advanced options dropdown with upload file option.
    with st.expander("Advanced Options", expanded=True):
        cmte_ex = st.checkbox("Advisory mode - *Consult a multidisciplinary panel for personalized insights and perspectives*")
        lib_ex = st.checkbox("Library mode - *Search trusted publications for authoritative answers*", value=True)
        doc_ex = st.checkbox("Upload Excel, PDF, or image file for examination")
        
    # If there's no openai api key, stop.
    if not openai_api_key:
        st.error("Please enter your OpenAI API key!")
        st.stop()
            
    # If Advisory mode was selected.
    if cmte_ex:
        # Create new form to query AI assistant.    
        with st.form(key="cmte_form", clear_on_submit=False):
            query = st.text_area("**Consult Advisors:**")
            submit = st.form_submit_button("Ask")
        # If submit button is clicked, query the aitam library.            
        if submit:
            # If form is submitted without a query, stop.
            if not query:
                st.error("Enter a question!")
                st.stop()            
            # Create new client for this submission.
            # client3 = OpenAI(api_key=openai_api_key)
            # client3 = AsyncOpenAI(api_key=openai_api_key)
            # Query the aitam library vector store and include internet
            # serach results.
            # Set up OpenAI client with your API key
            with st.spinner('Calculating...'):
                # event_loop = asyncio.get_running_loop()
                # if event_loop.is_running():
                #     response3 = asyncio.create_task(generate_response_cmte(openai_api_key, VECTOR_STORE_ID, query))
                # else:
                response3 = asyncio.run(generate_response_cmte(model, VECTOR_STORE_ID, query))
            st.write("*The insights provided reflect expert perspectives but are not a substitute for professional advice. Please consult legal, law enforcement, or threat management professionals before making decisions.*")
            st.markdown("#### Response")
            st.markdown(response3.final_output)
            # st.markdown(response3.messages[-1]['content'])
            # report all properties of the object
            # for method in dir(response3):
            #     # check if callable
            #     if callable(getattr(response3, method)):
            #         st.markdown(method) 
            # Write response.    
            # st.markdown(response3.choices[0].message)
            # for m in response3:
            #     st.markdown(m.content[0].text.value)
            # st.markdown(response3.choices[0].text)
            # st.markdown(response3.output[1].content[0].text)

    # If Library mode was selected.
    if lib_ex:
        # Create new form to search aitam library vector store.    
        with st.form(key="qa_form", clear_on_submit=False):
            query = st.text_area("**Search Library Holdings:**")
            submit = st.form_submit_button("Search")
        # If submit button is clicked, query the aitam library.            
        if submit:
            # If form is submitted without a query, stop.
            if not query:
                st.error("Enter a question to search the library!")
                st.stop()            
            # Setup output columns to display results.
            answer_col, sources_col = st.columns(2)
            # Create new client for this submission.
            client2 = OpenAI(api_key=openai_api_key)
            # Query the aitam library vector store and include internet
            # serach results.
            with st.spinner('Calculating...'):
                response2 = client2.responses.create(
                    input = query,
                    model = model,
                    temperature = 0.3,
                    tools = [{
                                "type": "file_search",
                                "vector_store_ids": [VECTOR_STORE_ID],
                    }],
                    include=["output[*].file_search_call.search_results"]
                )
            # Write response to the answer column.    
            with answer_col:
                st.write("*Information is drawn from published sources and academic literature. For critical decisions, consult qualified legal, law enforcement, or threat professionals.*")
                cleaned_response = re.sub(r'【.*?†.*?】', '', response2.output[1].content[0].text)
                st.markdown("#### Response")
                st.markdown(cleaned_response)
            # Write files used to generate the answer.
            with sources_col:            
                st.markdown("#### Sources")
                # Extract annotations from the response, and print source files.
                annotations = response2.output[1].content[0].annotations
                retrieved_files = set([response2.filename for response2 in annotations])
                file_list_str = ", ".join(retrieved_files)
                st.markdown(f"**File(s):** {file_list_str}")
    
                st.markdown("#### Token Usage")
                input_tokens = response2.usage.input_tokens
                output_tokens = response2.usage.output_tokens
                total_tokens = input_tokens + output_tokens
                input_tokens_str = f"{input_tokens:,}"
                output_tokens_str = f"{output_tokens:,}"
                total_tokens_str = f"{total_tokens:,}"
    
                st.markdown(
                    f"""
                    <p style="margin-bottom:0;">Input Tokens: {input_tokens_str}</p>
                    <p style="margin-bottom:0;">Output Tokens: {output_tokens_str}</p>
                    """,
                    unsafe_allow_html=True
                )
                st.markdown(f"Total Tokens: {total_tokens_str}")
    
                if model == "gpt-4.1-nano":
                    input_token_cost = .1/1000000
                    output_token_cost = .4/1000000
                elif model == "gpt-4o-mini":
                    input_token_cost = .15/1000000
                    output_token_cost = .6/1000000
                elif model == "gpt-4.1":
                    input_token_cost = 2.00/1000000
                    output_token_cost = 8.00/1000000
                elif model == "gpt-4.1-mini":
                    input_token_cost = .4/1000000
                    output_token_cost = 1.60/1000000
                elif model == "o4-mini":
                    input_token_cost = 1.10/1000000
                    output_token_cost = 4.40/1000000
    
                cost = input_tokens*input_token_cost + output_tokens*output_token_cost
                formatted_cost = "${:,.4f}".format(cost)
                
                st.markdown(f"**Total Cost:** {formatted_cost}")                
                
                # st.markdown("#### Sources")
                # # Extract annotations from the response, and print source files.
                # annotations = response2.output[1].content[0].annotations
                # retrieved_files = set([response2.filename for response2 in annotations])
                # file_list_str = ", ".join(retrieved_files)
                # st.markdown(f"**File(s):** {file_list_str}")
    
                # st.markdown("#### Token Usage")
                # input_tokens = response2.usage.input_tokens
                # output_tokens = response2.usage.output_tokens
                # total_tokens = input_tokens + output_tokens
                # input_tokens_str = f"{input_tokens:,}"
                # output_tokens_str = f"{output_tokens:,}"
                # total_tokens_str = f"{total_tokens:,}"
    
                # st.markdown(
                #     f"""
                #     <p style="margin-bottom:0;">Input Tokens: {input_tokens_str}</p>
                #     <p style="margin-bottom:0;">Output Tokens: {output_tokens_str}</p>
                #     """,
                #     unsafe_allow_html=True
                # )
                # # st.markdown(f"Input Tokens: {input_tokens}")
                # # st.markdown(f"Output Tokens: {output_tokens}")
                # st.markdown(f"Total Tokens: {total_tokens}")
    
                # cost = input_tokens*.1/1000000 + output_tokens*.4/1000000
                # formatted_cost = "${:,.4f}".format(cost)
                
                # st.markdown(f"**Total Cost:** {formatted_cost}")

    # If the option to upload a document was selected, allow for an upload and then 
    # process it.
    if doc_ex:
        # File uploader for Excel files
        uploaded_file = st.file_uploader("Choose an Excel, PDF, or image (heif, jpg, png) file", type=["xlsx","pdf","heif","jpg","png"], key="uploaded_file")
        # If a file is uploaded, extract the text and write serialized information to a text file, 
        # give options for further processing, and run assistant to process the information.
        if uploaded_file:
            # Read file, for each row combine column information, create json string, and
            # serialize the data for later processing by the openai model.
            if Path(uploaded_file.name).suffix.lower() == ".xlsx":            
                filename = extract_text_from_excel(uploaded_file)
            elif Path(uploaded_file.name).suffix.lower() == ".pdf":
                filename = copy_pdf(uploaded_file)
            elif Path(uploaded_file.name).suffix.lower() == ".heif" or Path(uploaded_file.name).suffix.lower() == ".jpg" or Path(uploaded_file.name).suffix.lower() == ".png" or Path(uploaded_file.name).suffix.lower() == ".jpeg":
                filename = convert_image_to_pdf(uploaded_file)
            # If there's no openai api key, stop.
            if not openai_api_key:
                st.error("Please enter your OpenAI API key!")
                st.stop()    
            # Form input and query
            with st.form("doc_form", clear_on_submit=False):
                # Create form to process file with the aitam assistant and be able to ask specific
                # questions about the file.
                submit_doc_ex = st.form_submit_button("Standard Examination", on_click=disable_button)
                query_doc_ex = st.text_area("**Custom Queries**")
                submit_doc_ex_form = st.form_submit_button("Submit Query")
                # If there's no openai api key, stop.
                if not openai_api_key:
                    st.error("Please enter your OpenAI API key!")
                    st.stop()
                # Conduct standard aitam eval on the file.
                if submit_doc_ex and doc_ex:
                    query_text = "I need your help analyzing the uploaded document."
                    # Call function to copy file to openai storage, create vector store, and use an 
                    # assistant to eval the file.
                    with st.spinner('Calculating...'):
                        (response, TMP_FILE_ID, TMP_VECTOR_STORE_ID, client, run, thread) = generate_response(filename, openai_api_key, model, MATH_ASSISTANT_ID, query_text)
                    # Write disclaimer and response from assistant eval of file.
                    st.write("*As the Threat AI system continues to be refined. Users should review the original file and verify the summary for reliability and relevance.*")
                    st.write("#### Summary")
                    i = 0
                    for m in response:
                        if i > 0:
                            st.markdown(m.content[0].text.value)
                        i += 1
                    # Reset the button state for standard aitam file eval, and 
                    # delete the file from openai storage and the associated
                    # vector store.
                    submit_doc_ex = False
                    delete_vectors(client, TMP_FILE_ID, TMP_VECTOR_STORE_ID)
                # If the user provides a custom query for the file and submits it, 
                # call different function to use a different assistant to run the 
                # query on the file.
                if submit_doc_ex_form:                    
                    with st.spinner('Calculating...'):
                        (response, TMP_FILE_ID, TMP_VECTOR_STORE_ID, client) = generate_response_noassist("temp.txt", openai_api_key, model, query_doc_ex)
                    # Write disclaimer and response from assistant eval of file.            
                    st.write("*As the Threat AI system continues to be refined. Users should review the original file and verify the summary for reliability and relevance.*")
                    for m in response:
                        st.markdown(m.content[0].text.value)
                    # Reset the button state for the custom aitam file eval, and 
                    # delete the file from openai storage and the associated
                    # vector store.            
                    submit_doc_ex_form = False
                    delete_vectors(client, TMP_FILE_ID, TMP_VECTOR_STORE_ID)

elif st.session_state.get('authentication_status') is False:
    st.error('Username/password is incorrect')

elif st.session_state.get('authentication_status') is None:
    st.warning('Please enter your username and password')

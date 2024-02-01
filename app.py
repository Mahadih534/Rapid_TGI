import gradio as gr
from huggingface_hub import InferenceClient

client = InferenceClient("mistralai/Mixtral-8x7B-Instruct-v0.1")

def format_prompt(message, history):
  prompt = "<s>"
  for user_prompt, bot_response in history:
    prompt += f"[INST] {user_prompt} [/INST]"
    prompt += f" {bot_response}</s> "
  prompt += f"[INST] {message} [/INST]"
  return prompt


description="""The Rapid TGI (Text Generation Inference) has developed by learning purpose 
                            <h3>Source Code:</h3>
                            <ul><li><a id='link' href='https://github.com/Mahadih534/Rapid_TGI'>Github Repository</a></li>
                            <li><a id='link' href='https://colab.research.google.com/drive/1Ti6Dn8F9GWozjAkjj0lHX3IskEVoOK5R'>Colab Link</a></li></ul>"""


title="<span id='logo'></span>"" Rapid TGI"

css="""
          .gradio-container {
              background: rgb(131,58,180);
              background: linear-gradient(90deg, rgba(131,58,180,1) 0%, rgba(253,29,29,1) 50%, rgba(252,176,69,1) 100%);

              #logo {
              content: url('https://i.ibb.co/6vz9WjL/chat-bot.png');
              width: 42px;
              height: 42px;
              margin-right: 10px;
              margin-top: 3px;
              display:inline-block;
            };

            #link {
            color: #fff;
            background-color: transparent;
            };
          }
          """

def inference(message, history, model="mistralai/Mixtral-8x7B-Instruct-v0.1", Temperature=0.3, tokens=512,top_p=0.95, r_p=0.93):

    Temperature = float(Temperature)
    if Temperature < 1e-2:
        Temperature = 1e-2
    top_p = float(top_p)

    kwargs = dict(
        temperature=Temperature,
        max_new_tokens=tokens,
        top_p=top_p,
        repetition_penalty=r_p,
        do_sample=True,
        seed=42,
    )
    prompt = format_prompt(message, history)
    client = InferenceClient(model=model)
    partial_message = ""
    for response in client.text_generation(prompt,**kwargs, stream=True, details=True, return_full_text=False):
        partial_message += response.token.text
        yield partial_message


chatbot = gr.Chatbot(avatar_images=["https://i.ibb.co/kGd6XrM/user.png", "https://i.ibb.co/6vz9WjL/chat-bot.png"], 
                     bubble_full_width=False, show_label=False, show_copy_button=True, likeable=True,)


UI=  gr.ChatInterface(
        inference,
        chatbot=chatbot,
        description=description,
        title=title,
        additional_inputs_accordion=gr.Accordion(label="Additional Configuration to get better response",open=False),
        retry_btn="Retry Again",
        undo_btn="Undo",
        clear_btn="Clear",
        theme="darksoft",
        submit_btn="Send",
        css=css,
        additional_inputs=[
                                gr.Dropdown(value="mistralai/Mixtral-8x7B-Instruct-v0.1",
                                  choices =["mistralai/Mixtral-8x7B-Instruct-v0.1","HuggingFaceH4/zephyr-7b-beta",
                                    "mistralai/Mistral-7B-Instruct-v0.1"], label="Available models",
                                    info="default model is Mixtral-8x7B-Instruct-v0.1",interactive=True,),
                                gr.Slider(value=0.3, maximum=1.0,label="Temperature"),
                                gr.Slider(value=512, maximum=1020,label="Max New Tokens"),
                                gr.Slider(value=0.95, maximum=1.0,label="Top P"),
                                gr.Slider(value=0.93, maximum=1.0,label="Repetition Penalty"),
                            ],
        examples=[["Hello"], ["can i know about generative ai ?"], ["how can i deploy a LLM in hugguingface inference endpoint ?"]],
        
    )
UI.queue().launch(show_api= False,max_threads=50)

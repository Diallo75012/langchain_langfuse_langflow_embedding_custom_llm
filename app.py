import os
from langfuse import Langfuse
from langfuse.decorators import observe, langfuse_context
from dotenv import load_dotenv


load_dotenv()

langfuse = Langfuse(
  secret_key=os.gentenv("LANG_SECRET_KEY"),
  public_key=os.gentenv("LANG_PUBLIC_KEY"),
  host=os.getenv("LANG_HOST")
)

# Wrap LLM function with decorator to get any llm observed through langfuse: @observe(as_type="generation")
# example function optional, extract some fields from kwargs
# update observation
"""
# update observation
  kwargs_clone = kwargs.copy()
  input = kwargs_clone.pop('messages', None)
  model = kwargs_clone.pop('model', None)
  langfuse_context.update_current_observation(
      input=input,
      model=model,
      metadata=kwargs_clone
  )
"""
# update trace/observation
"""
# update trace attributes (e.g, name, session_id, user_id)
    langfuse_context.update_current_trace(
        name="custom-trace",
        session_id="user-1234",
        user_id="session-1234",
    )
# get the langchain handler for the current trace
  langfuse_handler = langfuse_context.get_current_langchain_handler()

  ...
  # Your Langchain code
  ...

  # Add Langfuse handler as callback (classic and LCEL)
  chain.invoke({"input": "<user_input>"}, config={"callbacks": [langfuse_handler]})

"""

# get get the URL of the current trace using 
"""
langfuse_context.get_current_trace_url()
"""

# Get trace and observation IDs 
"""
langfuse_context.get_current_trace_id()
langfuse_context.get_current_observation_id()
"""

# enrish the observation
# get traces sanitized captured. So here you don't want to show raw imput/output so you have function or a system to sanitize those and trace those
# therefore, have control on output/input by deactivating first the @observe() default behavior which is to capture all
"""
 langfuse_context.update_current_observation(
        input="sanitized input", # any serializable object
        output="sanitized output", # any serializable object
    )
"""

# for langchain app
"""
# Initialize Langfuse handler
from langfuse.callback import CallbackHandler
langfuse_handler = CallbackHandler(
    secret_key="sk-lf-...",
    public_key="pk-lf-...",
    host="https://cloud.langfuse.com", # just put localhost here probably
)

# Your Langchain code

# Add Langfuse handler as callback (classic and LCEL) to you invokation or runs (invoke(), predict(), run(), call()):
chain.invoke({"input": "<user_input>"}, config={"callbacks": [langfuse_handler]})
chain.run(input="<user_input>", callbacks=[langfuse_handler])
conversation.predict(input="<user_input>", callbacks=[langfuse_handler])
"""

# traces are processed on the background to have all of those tasls treated and not lost when the application stops you can use the  '.flush()'.
# It is blocking so like a graceful stop of the app with traces tasks done and recorded properly.
"""
langfuse_context.flush()
"""





















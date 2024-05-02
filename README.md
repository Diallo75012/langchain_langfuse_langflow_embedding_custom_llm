# this repo will be using lanflow and langfuse to trace embedding and llm interaction constructing a flow of logic in order to get some job done by ai agents
# I believe that mini ai agents will be future for people who want to have automated task done at a micro level and to replace some human tasks by decomposing what make
# those job position or flow of tasks and getting mini agents fulfilling the requests to completely automate the process.

# project stack (all still under construction)
# langchain, langflow, lmstudio with some models (mistral7B), ollama (mistral7b), groq (llama3 8b, 70b. mixtral8x7b), langchain, postgresql16, pgvector
# clone repo
```bash
git clone <this_repo>.git
```
# create a .env file and put all envrionment variables in
```bash
nano .env
...
...
```
# create virtual environment and install requirements
```python
python3 -m venv <name_you_want>_venv
# start virtual envrionment
source <name_of_python_environment>_venv/bin.activate
# install all requirements
pip install -r requirements.txt
```

# start langfuse
```bash
cd langfuse/
docker compose up
# then just go to the local url: http://localhost:3000
```

# start langflow
```bash
langflow run
```
